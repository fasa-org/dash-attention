import torch
import triton
import triton.language as tl
from native_sparse_attention.ops.triton.utils import get_num_warps_stages, is_hopper_gpu

IS_HOPPER_GPU = is_hopper_gpu()

@triton.jit
def dq_kernel(
    q_ptr, k_ptr, ds_ptr, dq_ptr,
    kernel_size, kernel_stride,
    cu_seqlens_q, cu_seqlens_k,
    NUM_KV_HEADS: tl.constexpr, NUM_SHARE_Q_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
    sm_scale,
    stride_qn, stride_qh, stride_qd,
    stride_kn, stride_kh, stride_kd,
    stride_dsh, stride_dsn, stride_dsk,  
    stride_dqn, stride_dqh, stride_dqd,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_qb = tl.program_id(2)

    pid_kh = pid_h // NUM_SHARE_Q_HEADS

    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len   = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    k_start = tl.load(cu_seqlens_k + pid_b)
    k_len   = tl.load(cu_seqlens_k + pid_b + 1) - k_start

    # scale = sm_scale * 1.44269504 # do not need log2(e) here
    scale = sm_scale

    q0 = pid_qb * BLOCK_Q + kernel_size - 1
    if q0 >= q_len:
        return

    off_q = q0 + tl.arange(0, BLOCK_Q)
    off_d = tl.arange(0, BLOCK_D)

    dq = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

    hi = (q0 + BLOCK_Q - kernel_size) // kernel_stride + 1
    hi = tl.minimum(hi, k_len)

    for kb in range(0, hi, BLOCK_K):
        off_k = kb + tl.arange(0, BLOCK_K)                  
        key_pos = off_k * kernel_stride + (kernel_size - 1)

        ds_ptrs = (
            ds_ptr
            + pid_h * stride_dsh
            + (q_start + off_q)[:, None] * stride_dsn
            + off_k[None, :] * stride_dsk
        )
        vis = off_q[:, None] >= key_pos[None, :]
        valid_q = off_q >= (kernel_size - 1)
        # ds = tl.load(
        #     ds_ptrs,
        #     mask=(off_q[:, None] < q_len)
        #          & (off_k[None, :] < k_len)
        #          & valid_q[:, None]
        #          & vis,
        #     other=0.0
        # ).to(tl.float32)

        reach = off_k[None, :] < hi 
        ds = tl.load(
            ds_ptrs,
            mask=(off_q[:, None] < q_len)
                & (off_k[None, :] < k_len)
                & reach
                & valid_q[:, None]
                & vis,
            other=0.0
        ).to(tl.float32)

        # load K (packed global, 所以用 k_start+off_k)
        k_ptrs = (
            k_ptr
            + (k_start + off_k)[:, None] * stride_kn
            + pid_kh * stride_kh
            + off_d[None, :] * stride_kd
        )
        k = tl.load(
            k_ptrs,
            mask=(off_k[:, None] < k_len) & (off_d[None, :] < HEAD_DIM),
            other=0.0
        ).to(tl.float32)

        dq += tl.dot(ds, k) * scale

    dq_ptrs = (
        dq_ptr
        + (q_start + off_q)[:, None] * stride_dqn
        + pid_h * stride_dqh
        + off_d[None, :] * stride_dqd
    )
    tl.store(dq_ptrs, dq.to(tl.bfloat16), mask=(off_q[:, None] < q_len) & (off_d[None, :] < HEAD_DIM))

@triton.jit
def dk_kernel(
    q_ptr, ds_ptr, dk_ptr,
    kernel_size, kernel_stride,
    cu_seqlens_q, cu_seqlens_k,
    NUM_KV_HEADS: tl.constexpr,
    NUM_SHARE_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    sm_scale,
    stride_qn, stride_qh, stride_qd,
    stride_dsh, stride_dsn, stride_dsk,     # dS: [Hq, total_q, max_seqlen_k]  (K dim starts from 0 per-batch)
    stride_dkn, stride_dkh, stride_dkd,     # dK: [total_k, Hk, D]
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b  = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_kb = tl.program_id(2)

    # batch ranges
    q_start = tl.load(cu_seqlens_q + pid_b).to(tl.int32)
    q_end   = tl.load(cu_seqlens_q + pid_b + 1).to(tl.int32)
    k_start = tl.load(cu_seqlens_k + pid_b).to(tl.int32)
    k_end   = tl.load(cu_seqlens_k + pid_b + 1).to(tl.int32)

    q_len = q_end - q_start
    k_len = k_end - k_start

    # key block indices within THIS batch
    off_k = pid_kb * BLOCK_K + tl.arange(0, BLOCK_K)
    if pid_kb * BLOCK_K >= k_len:
        return

    off_d = tl.arange(0, BLOCK_D)

    # key position mapping (must match forward mask)
    key_pos = off_k * kernel_stride + (kernel_size - 1)  # [BLOCK_K]

    # scale = sm_scale * 1.44269504  # same as forward
    scale = sm_scale

    dK = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)

    # q heads sharing this kv head
    hq0 = pid_kh * NUM_SHARE_Q_HEADS
    for hh in range(0, NUM_SHARE_Q_HEADS):
        pid_h = hq0 + hh  # q head id

        # iterate q blocks in this batch
        for qb in range(0, q_len, BLOCK_Q):
            off_q = qb + tl.arange(0, BLOCK_Q)  # batch-local q positions

            # forward: q_start_in_seq = pid_q*BLOCK_Q + (kernel_size - 1)
            # and it skips blocks where q_start_in_seq >= q_len.
            q0 = qb + (kernel_size - 1)
            # compute forward's hi bound for THIS q-block
            hi = (q0 + BLOCK_Q - kernel_size) // kernel_stride + 1
            hi = tl.maximum(hi, 0)
            hi = tl.minimum(hi, k_len)

            # only keys < hi were ever written by forward for this q-block
            reach = off_k[None, :] < hi

            valid_q = off_q >= (kernel_size - 1)
            vis = off_q[:, None] >= key_pos[None, :]

            # load dS: K dim is 0..k_len-1 (no k_start offset!)
            ds_ptrs = (
                ds_ptr
                + pid_h * stride_dsh
                + (q_start + off_q)[:, None] * stride_dsn
                + off_k[None, :] * stride_dsk
            )
            ds = tl.load(
                ds_ptrs,
                mask=(off_q[:, None] < q_len)
                     & (off_k[None, :] < k_len)
                     & valid_q[:, None]
                     & vis
                     & reach,
                other=0.0
            )
            # keep ds in bf16/fp16 for tensorcore dot; accumulate in fp32
            ds = ds.to(tl.bfloat16)

            # load Q
            q_ptrs = (
                q_ptr
                + (q_start + off_q)[:, None] * stride_qn
                + pid_h * stride_qh
                + off_d[None, :] * stride_qd
            )
            q = tl.load(
                q_ptrs,
                mask=(off_q[:, None] < q_len) & (off_d[None, :] < HEAD_DIM),
                other=0.0
            ).to(tl.bfloat16)

            # dK += dS^T @ Q
            dK += tl.dot(tl.trans(ds), q).to(tl.float32) * scale

    # store dK to packed global K (needs k_start offset!)
    dk_ptrs = (
        dk_ptr
        + (k_start + off_k)[:, None] * stride_dkn
        + pid_kh * stride_dkh
        + off_d[None, :] * stride_dkd
    )
    tl.store(
        dk_ptrs,
        dK.to(tl.bfloat16),
        mask=(off_k[:, None] < k_len) & (off_d[None, :] < HEAD_DIM)
    )

def compressed_attention_bwd(dS, q, k, kernel_size, kernel_stride, cu_seqlens_q, cu_seqlens_k, sm_scale):

    # if dS.isnan().any():
    #     rank = torch.distributed.get_rank()
    #     print(f"rank: {rank}")
    #     if rank == 22:
    #         breakpoint()
    # torch.distributed.barrier()


    q_len, q_heads, d = q.shape
    k_len, kv_heads, _ = k.shape
    num_share = q_heads // kv_heads

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)

    BLOCK_Q = 128
    BLOCK_K = 128
    BLOCK_D = triton.next_power_of_2(d)

    # dQ
    grid_dq = (cu_seqlens_q.numel() - 1, q_heads, triton.cdiv(q_len, BLOCK_Q))
    dq_kernel[grid_dq](
        q, k, dS, dq,
        kernel_size, kernel_stride,
        cu_seqlens_q, cu_seqlens_k,
        kv_heads, num_share, d, sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        dS.stride(0), dS.stride(1), dS.stride(2),
        dq.stride(0), dq.stride(1), dq.stride(2),
        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )

    # dK
    grid_dk = (cu_seqlens_k.numel() - 1, kv_heads, triton.cdiv(k_len, BLOCK_K))
    dk_kernel[grid_dk](
        q, dS, dk,
        kernel_size, kernel_stride,
        cu_seqlens_q, cu_seqlens_k,
        kv_heads, num_share, d, sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        dS.stride(0), dS.stride(1), dS.stride(2),
        dk.stride(0), dk.stride(1), dk.stride(2),
        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )
    # breakpoint()
    # if dk.isnan().any() or dk.isinf().any():
    #     breakpoint()
    # torch.distributed.barrier()
    return dq, dk