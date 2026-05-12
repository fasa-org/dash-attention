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
    """Find the nth set bit (1-indexed) in mask. Returns bit position (0-31)."""
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
def _masked_fa3_fwd_kernel(
    Q, K, V, BMASK, CHUNK_PRIOR, LEFT_PAD_OFFSETS,
    O_PARTIAL, L_PARTIAL, M_PARTIAL,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_mb, stride_mh, stride_mi,
    stride_cpb, stride_cph, stride_cpn,
    stride_ob, stride_oh, stride_os,
    stride_lb, stride_lh, stride_ls,
    scale,
    SEQ_LEN,
    N_INTS: tl.constexpr,
    N_INTS_PER_SPLIT: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    D_HEAD: tl.constexpr,
    HQ_RATIO: tl.constexpr,
    HKV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HAS_ATTN_MASK: tl.constexpr,
):
    pid_split = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch_id = pid_bh // HKV
    kv_head_id = pid_bh % HKV

    left_pad_offset = tl.load(LEFT_PAD_OFFSETS + batch_id) if HAS_ATTN_MASK else 0

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D_HEAD)
    offs_n = tl.arange(0, CHUNK_SIZE)

    # Load Q [BLOCK_M, D_HEAD], pad extra rows for tensor cores
    q_offset = batch_id * stride_qb + kv_head_id * HQ_RATIO * stride_qh
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < HQ_RATIO, other=0.0)

    # Accumulators in fp32
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)

    # Base pointers for this (batch, kv_head)
    k_base = K + batch_id * stride_kb + kv_head_id * stride_kh
    v_base = V + batch_id * stride_vb + kv_head_id * stride_vh
    bmask_ptr = BMASK + batch_id * stride_mb + kv_head_id * stride_mh

    int_start = pid_split * N_INTS_PER_SPLIT

    for n_int_offset in tl.range(N_INTS_PER_SPLIT):
        n_int = int_start + n_int_offset
        if n_int < N_INTS:
            bmask_val = tl.load(bmask_ptr + n_int * stride_mi).to(tl.int32, bitcast=True)
            # n_active = _popc(bmask_val)
            base_block = n_int * 32

            # Check later on if it's worth bringing this back.
            # for i_blk in tl.range(n_active):
            #     bit_pos = _fns(bmask_val, i_blk + 1)
            #     c_block = base_block + bit_pos
            for i in tl.static_range(32):
                if (bmask_val >> i) & 1:
                    c_block = base_block + i
                    chunk_start = c_block * CHUNK_SIZE

                    # Load K and V together before compute [CHUNK_SIZE, D_HEAD]
                    k_ptrs = k_base + (chunk_start + offs_n[:, None]) * stride_kn + offs_d[None, :] * stride_kd
                    v_ptrs = v_base + (chunk_start + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vd
                    tok = chunk_start + offs_n
                    kv_mask = (tok[:, None] >= left_pad_offset) & (tok[:, None] < SEQ_LEN)
                    k = tl.load(k_ptrs, mask=kv_mask, other=0.0, eviction_policy="evict_first")
                    v = tl.load(v_ptrs, mask=kv_mask, other=0.0, eviction_policy="evict_first")

                    # QK^T: [BLOCK_M, CHUNK_SIZE]
                    qk = tl.dot(q, tl.trans(k), input_precision="ieee").to(tl.float32) * scale

                    # Add log(chunk_prior) bias for this chunk
                    log_prior_c = tl.log(
                        tl.load(CHUNK_PRIOR + batch_id * stride_cpb + kv_head_id * stride_cph + c_block * stride_cpn).to(tl.float32)
                    )
                    qk = qk + log_prior_c

                    # Mask left-pad and out-of-bounds positions
                    qk = tl.where(
                        (tok[None, :] >= left_pad_offset) & (tok[None, :] < SEQ_LEN),
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

                    o_i += tl.dot(p.to(q.dtype), v, input_precision="ieee").to(tl.float32)

                    m_i = m_new

    # Store partial results (only first HQ_RATIO rows)
    hq_offset = kv_head_id * HQ_RATIO
    valid_mask = offs_m < HQ_RATIO

    o_ptrs = (O_PARTIAL + batch_id * stride_ob
              + (hq_offset + offs_m[:, None]) * stride_oh
              + pid_split * stride_os + offs_d[None, :])
    tl.store(o_ptrs, o_i, mask=offs_m[:, None] < HQ_RATIO)

    lm_ptrs = batch_id * stride_lb + (hq_offset + offs_m) * stride_lh + pid_split * stride_ls
    tl.store(L_PARTIAL + lm_ptrs, l_i, mask=valid_mask)
    tl.store(M_PARTIAL + lm_ptrs, m_i, mask=valid_mask)

@triton.autotune(
    configs=get_configs(),
    key=["D_HEAD"],
)
@triton.jit
def _masked_fa3_reduce_kernel(
    O_PARTIAL, L_PARTIAL, M_PARTIAL, O,
    stride_ob, stride_oh, stride_os,
    stride_lb, stride_lh, stride_ls,
    stride_rob, stride_roh,
    HQ: tl.constexpr,
    D_HEAD: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_id = pid // HQ
    hq_id = pid % HQ

    offs_d = tl.arange(0, D_HEAD)

    partial_base = batch_id * stride_ob + hq_id * stride_oh
    lm_base = batch_id * stride_lb + hq_id * stride_lh

    # Online reduction across splits.
    # Splits with no attended chunks store m=-inf, l=0, o=0. Two NaN traps:
    #   (a) when both m_acc and m_s are -inf (first split empty), m_new=-inf
    #       and tl.exp(-inf - -inf) = exp(NaN) = NaN, poisoning l/o forever.
    #   (b) when ALL splits are empty, l_acc=0 → 0/0 NaN at the final divide.
    # Guard each alpha so a side with effectively-empty m contributes 0;
    # guard the final divide. Threshold is well below any legitimate attention
    # score (attention scores are bounded by |q|*|k|/sqrt(d), realistically <<1e10).
    EMPTY_M: tl.constexpr = -1.0e15

    m_acc = float('-inf')
    l_acc = 0.0
    o_acc = tl.zeros([D_HEAD], dtype=tl.float32)

    for s in range(NUM_SPLITS):
        m_s = tl.load(M_PARTIAL + lm_base + s * stride_ls)
        l_s = tl.load(L_PARTIAL + lm_base + s * stride_ls)
        o_s = tl.load(O_PARTIAL + partial_base + s * stride_os + offs_d)

        m_new = tl.maximum(m_acc, m_s)
        alpha_acc = tl.where(m_acc < EMPTY_M, 0.0, tl.exp(m_acc - m_new))
        alpha_s = tl.where(m_s < EMPTY_M, 0.0, tl.exp(m_s - m_new))

        l_acc = l_acc * alpha_acc + l_s * alpha_s
        o_acc = o_acc * alpha_acc + o_s * alpha_s
        m_acc = m_new

    o_acc = tl.where(l_acc > 0, o_acc / l_acc, 0.0)

    o_ptr = O + batch_id * stride_rob + hq_id * stride_roh + offs_d
    tl.store(o_ptr, o_acc)


def masked_fa3(q, k, v, bmask, chunk_size, attn_mask=None, chunk_prior=None):
    """
    Masked flash attention decoding kernel with split-KV and left-padding support.

    Args:
        q: [B, Hq, 1, D] queries (bf16/fp16)
        k: [B, Hkv, N, D] KV cache keys
        v: [B, Hkv, N, D] KV cache values
        bmask: [B, Hkv, 1, n_ints] bit-packed block mask (int32)
        chunk_size: KV block size
        attn_mask: [B, N] bool, True = real token, False = left-pad token (defaults to all True)

    Returns:
        [B, Hq, 1, D] attention output (fp32)
    """
    B, Hq, _, D = q.shape
    _, Hkv, N, _ = k.shape
    HQ_RATIO = Hq // Hkv
    n_chunks = math.ceil(N / chunk_size)
    n_ints = math.ceil(n_chunks / 32)

    # split kv
    num_splits = max(1, min(n_ints, 128 // max(1, B * Hkv)))
    n_ints_per_split = math.ceil(n_ints / num_splits)
    num_splits = math.ceil(n_ints / n_ints_per_split)

    has_attn_mask = attn_mask is not None
    if has_attn_mask:
        left_pad_offsets = (N - attn_mask.sum(dim=-1)).to(torch.int32).to(q.device)
    else:
        left_pad_offsets = q  # dummy, not accessed by kernel when HAS_ATTN_MASK=False

    chunk_prior_flat = chunk_prior.squeeze(2).contiguous()  # [B, Hkv, n_chunks]

    BLOCK_M = max(triton.next_power_of_2(HQ_RATIO), 16)
    scale = 1.0 / math.sqrt(D)

    # Intermediate buffers: layout [B, Hq, num_splits, ...] for contiguous reduction reads
    # L_partial and M_partial are fully written by the fwd kernel (unconditional stores at end
    # of each TB initialised to values 0 / -inf), so torch.empty is safe here (i think)
    O_partial = torch.empty(B, Hq, num_splits, D, device=q.device, dtype=torch.float32)
    L_partial = torch.empty(B, Hq, num_splits, device=q.device, dtype=torch.float32)
    M_partial = torch.empty(B, Hq, num_splits, device=q.device, dtype=torch.float32)
    O = torch.empty(B, Hq, 1, D, device=q.device, dtype=torch.float32)

    _masked_fa3_fwd_kernel[(num_splits, B * Hkv)](
        q, k, v, bmask, chunk_prior_flat, left_pad_offsets,
        O_partial, L_partial, M_partial,
        q.stride(0), q.stride(1), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        bmask.stride(0), bmask.stride(1), bmask.stride(3),
        chunk_prior_flat.stride(0), chunk_prior_flat.stride(1), chunk_prior_flat.stride(2),
        O_partial.stride(0), O_partial.stride(1), O_partial.stride(2),
        L_partial.stride(0), L_partial.stride(1), L_partial.stride(2),
        scale,
        SEQ_LEN=N,
        N_INTS=n_ints,
        N_INTS_PER_SPLIT=n_ints_per_split,
        CHUNK_SIZE=chunk_size,
        D_HEAD=D,
        HQ_RATIO=HQ_RATIO,
        HKV=Hkv,
        BLOCK_M=BLOCK_M,
        HAS_ATTN_MASK=has_attn_mask,
    )

    _masked_fa3_reduce_kernel[(B * Hq,)](
        O_partial, L_partial, M_partial, O,
        O_partial.stride(0), O_partial.stride(1), O_partial.stride(2),
        L_partial.stride(0), L_partial.stride(1), L_partial.stride(2),
        O.stride(0), O.stride(1),
        HQ=Hq,
        D_HEAD=D,
        NUM_SPLITS=num_splits,
    )

    return O
