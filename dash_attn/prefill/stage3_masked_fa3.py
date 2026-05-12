import triton
import triton.language as tl
import torch
import math


@triton.jit
def _popc(x):
    return tl.inline_asm_elementwise(
        "popc.b32 $0, $1;",
        "=r,r",
        [x],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fns(mask, nth):
    return tl.inline_asm_elementwise(
        "fns.b32 $0, $1, 0, $2;",
        "=r,r,r",
        [mask, nth],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )

def get_configs():
    return [
        triton.Config(
            {},
            num_warps=nw,
            num_stages=ns,
        )
        for nw in [1, 2, 4]
        for ns in [1, 2, 3, 4]
    ]
@triton.autotune(
    configs=get_configs(),
    key=["D_HEAD"],
)
@triton.jit
def _masked_fa3_prefill_fwd_kernel(
    Q, K, V, BMASK, CHUNK_PRIOR, LEFT_PAD_OFFSETS,
    O,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_mb, stride_mh, stride_ms, stride_mi,
    stride_cpb, stride_cph, stride_cps, stride_cpn,
    stride_ob, stride_oh, stride_os,
    scale,
    SEQ_LEN,
    N_INTS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    D_HEAD: tl.constexpr,
    HQ_RATIO: tl.constexpr,
    HKV: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_seq = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch_id = pid_bh // HKV
    kv_head_id = pid_bh % HKV

    left_pad_offset = tl.load(LEFT_PAD_OFFSETS + batch_id)

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D_HEAD)
    offs_n = tl.arange(0, CHUNK_SIZE)

    # Load Q [BLOCK_M, D_HEAD] for this sequence position across HQ_RATIO heads
    q_offset = batch_id * stride_qb + kv_head_id * HQ_RATIO * stride_qh + pid_seq * stride_qs
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < HQ_RATIO, other=0.0)

    # Accumulators in fp32
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)

    # Base pointers for this (batch, kv_head, seq_pos)
    k_base = K + batch_id * stride_kb + kv_head_id * stride_kh
    v_base = V + batch_id * stride_vb + kv_head_id * stride_vh
    bmask_ptr = BMASK + batch_id * stride_mb + kv_head_id * stride_mh + pid_seq * stride_ms
    cp_ptr = CHUNK_PRIOR + batch_id * stride_cpb + kv_head_id * stride_cph + pid_seq * stride_cps

    # Diagonal chunk: the chunk that contains pid_seq. Keys beyond pid_seq in this chunk
    # are future tokens that must be masked out for causality.
    diag_chunk = pid_seq // CHUNK_SIZE

    for n_int in range(N_INTS):
        bmask_val = tl.load(bmask_ptr + n_int * stride_mi).to(tl.int32, bitcast=True)
        n_active = _popc(bmask_val)
        base_block = n_int * 32

        for i_blk in range(n_active):
            bit_pos = _fns(bmask_val, i_blk + 1)
            c_block = base_block + bit_pos
            chunk_start = c_block * CHUNK_SIZE
            tok = chunk_start + offs_n  # [CHUNK_SIZE]

            # Past chunks are fully valid; only the diagonal chunk needs causal trimming.
            causal_ok = (c_block < diag_chunk) | (tok <= pid_seq)

            # Load K [CHUNK_SIZE, D_HEAD]
            k_ptrs = k_base + (chunk_start + offs_n[:, None]) * stride_kn + offs_d[None, :] * stride_kd
            kv_mask = causal_ok[:, None] & (tok[:, None] >= left_pad_offset) & (tok[:, None] < SEQ_LEN)
            k = tl.load(k_ptrs, mask=kv_mask, other=0.0)

            # QK^T: [BLOCK_M, CHUNK_SIZE]
            qk = tl.dot(q, tl.trans(k), input_precision="ieee").to(tl.float32) * scale

            # Add log(chunk_prior) bias for this chunk
            log_prior_c = tl.log(
                tl.load(cp_ptr + c_block * stride_cpn).to(tl.float32)
            )
            qk = qk + log_prior_c

            # Mask left-pad, out-of-bounds, and future positions
            qk = tl.where(
                causal_ok[None, :] & (tok[None, :] >= left_pad_offset) & (tok[None, :] < SEQ_LEN),
                qk,
                float('-inf'),
            )

            # Online softmax update
            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            o_i = o_i * alpha[:, None]

            # Load V [CHUNK_SIZE, D_HEAD] and accumulate
            v_ptrs = v_base + (chunk_start + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=kv_mask, other=0.0)
            o_i += tl.dot(p.to(q.dtype), v, input_precision="ieee").to(tl.float32)

            m_i = m_new

    # Normalise; guard against NaN for left-padded query positions (l_i == 0)
    o_i = tl.where(l_i[:, None] > 0, o_i / l_i[:, None], 0.0)

    # Store [BLOCK_M, D_HEAD] directly, no reduce kernel needed
    hq_offset = kv_head_id * HQ_RATIO
    o_ptrs = (O
              + batch_id * stride_ob
              + (hq_offset + offs_m[:, None]) * stride_oh
              + pid_seq * stride_os
              + offs_d[None, :])
    tl.store(o_ptrs, o_i, mask=offs_m[:, None] < HQ_RATIO)


def masked_fa3(q, k, v, bmask, chunk_size, attn_mask=None, chunk_prior=None):
    """
    Masked flash attention prefill kernel with causal masking and left-padding support.

    Args:
        q: [B, Hq, N, D] queries (bf16/fp16)
        k: [B, Hkv, N, D] keys
        v: [B, Hkv, N, D] values
        bmask: [B, Hkv, N, n_ints] bit-packed block mask (int32); each query token has its own row
        chunk_size: KV block size
        attn_mask: [B, N] bool, True=real token, False=left-pad (defaults to all True)
        chunk_prior: [B, Hkv, N, n_chunks] float32 (defaults to all ones)

    Returns:
        [B, Hq, N, D] attention output (fp32)
    """
    B, Hq, N, D = q.shape
    _, Hkv, _, _ = k.shape
    HQ_RATIO = Hq // Hkv
    n_chunks = math.ceil(N / chunk_size)
    n_ints = math.ceil(n_chunks / 32)

    if attn_mask is not None:
        left_pad_offsets = (N - attn_mask.sum(dim=-1)).to(torch.int32).to(q.device)
    else:
        left_pad_offsets = torch.zeros(B, device=q.device, dtype=torch.int32)

    if chunk_prior is None:
        chunk_prior_flat = torch.ones(B, Hkv, N, n_chunks, device=q.device, dtype=torch.float32)
    else:
        chunk_prior_flat = chunk_prior.contiguous()  # [B, Hkv, N, n_chunks]

    BLOCK_M = max(triton.next_power_of_2(HQ_RATIO), 16)
    scale = 1.0 / math.sqrt(D)
    O = torch.empty(B, Hq, N, D, device=q.device, dtype=torch.float32)

    _masked_fa3_prefill_fwd_kernel[(N, B * Hkv)](
        q, k, v, bmask, chunk_prior_flat, left_pad_offsets,
        O,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        bmask.stride(0), bmask.stride(1), bmask.stride(2), bmask.stride(3),
        chunk_prior_flat.stride(0), chunk_prior_flat.stride(1),
        chunk_prior_flat.stride(2), chunk_prior_flat.stride(3),
        O.stride(0), O.stride(1), O.stride(2),
        scale,
        SEQ_LEN=N,
        N_INTS=n_ints,
        CHUNK_SIZE=chunk_size,
        D_HEAD=D,
        HQ_RATIO=HQ_RATIO,
        HKV=Hkv,
        BLOCK_M=BLOCK_M,
    )
    return O

