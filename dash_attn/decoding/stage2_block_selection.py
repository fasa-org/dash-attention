import math

import torch
import triton
import triton.language as tl

# Kernel 1
# --------
# Per (B, Hkv): tensor-core matmul of q [GROUPS_padded, D] against
# k_blk [BLOCK_N, D] tiles. Stores per-(B, Hq) row scores and per-row max.

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
    key=["D"],
)
@triton.jit
def _scores_and_max_kernel(
    Q, K_BLK, SCORES, MAXV, FIRST_VALID_CHUNK,
    n_chunks,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_sb, stride_sh, stride_sn,
    stride_mb, stride_mh,
    scale,
    GROUPS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_ATTN_MASK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hkv = tl.program_id(1)
    first_valid = tl.load(FIRST_VALID_CHUNK + pid_b) if HAS_ATTN_MASK else 0

    offs_m = tl.arange(0, BLOCK_M)
    valid_m = offs_m < GROUPS
    offs_d = tl.arange(0, D)
    hq_offs = pid_hkv * GROUPS + offs_m

    q = tl.load(
        Q + pid_b * stride_qb + hq_offs[:, None] * stride_qh + offs_d[None, :] * stride_qd,
        mask=valid_m[:, None],
        other=0.0,
    )

    k_ptr = K_BLK + pid_b * stride_kb + pid_hkv * stride_kh
    s_ptr = SCORES + pid_b * stride_sb

    NEG: tl.constexpr = -1.0e9
    max_val = tl.full((BLOCK_M,), NEG, dtype=tl.float32)

    n_iters = tl.cdiv(n_chunks, BLOCK_N)
    for i in range(0, n_iters):
        offs_n = i * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < n_chunks
        k = tl.load(
            k_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=mask_n[:, None],
            other=0.0,
        )
        scores_tile = tl.dot(q, tl.trans(k), input_precision="ieee").to(tl.float32) * scale
        chunk_valid = offs_n >= first_valid  # [BLOCK_N]
        scores_to_store = tl.where(chunk_valid[None, :], scores_tile, NEG)
        tl.store(
            s_ptr + hq_offs[:, None] * stride_sh + offs_n[None, :] * stride_sn,
            scores_to_store,
            mask=valid_m[:, None] & mask_n[None, :],
        )
        s_for_max = tl.where(mask_n[None, :] & chunk_valid[None, :], scores_tile, NEG)
        max_val = tl.maximum(max_val, tl.max(s_for_max, axis=1))

    tl.store(
        MAXV + pid_b * stride_mb + hq_offs * stride_mh,
        max_val,
        mask=valid_m,
    )


# Kernel 2
# --------
# Per (B, Hkv):
#
#   Phase A (per-row tau search): tile-loop over chunks, accumulate ff/df/ddf
#   sums per Hq row, then Halley update on tau. Repeat N_ITER times.
#
#   Phase B (streaming output): tile-loop over chunks in BLOCK_N=32 tiles.
#   Per tile, compute final entmax probs, mean across Hq heads in the group,
#   pack `mean_prob > 0` into one int32, store. Also writes prior fp32.
#   The trailing "current partial chunk" column at index n_chunks is appended
#   as prior=1.0 and bit=1.

@triton.autotune(
    configs=get_configs(),
    key=["GROUPS"],
)
@triton.jit
def _entmax_mean_pack_kernel(
    SCORES, MAXV, PRIOR, BMASK,
    n_chunks,
    stride_sb, stride_sh, stride_sn,
    stride_mb, stride_mh,
    stride_pb, stride_ph, stride_pn,
    stride_bb, stride_bh, stride_bi,
    n_ints,
    sigma,
    GROUPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_ITER: tl.constexpr,
    ESTIMATE_DIAG: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hkv = tl.program_id(1)

    EPS: tl.constexpr = 1.0e-5
    SCALAR: tl.constexpr = 0.5
    NEG_FILL: tl.constexpr = -1.0e4

    offs_m = tl.arange(0, BLOCK_M)
    valid_m = offs_m < GROUPS
    hq_offs = pid_hkv * GROUPS + offs_m

    s_base = SCORES + pid_b * stride_sb
    s_row_ptrs = s_base + hq_offs[:, None] * stride_sh

    max_val = tl.load(
        MAXV + pid_b * stride_mb + hq_offs * stride_mh,
        mask=valid_m,
        other=NEG_FILL,
    ).to(tl.float32)
    max_scaled = max_val * SCALAR

    # tau initialization
    t_hi = max_scaled
    t_lo = max_scaled - 1.0
    t = 0.5 * (t_lo + t_hi)

    n_tiles = tl.cdiv(n_chunks, BLOCK_N)

    # Phase A: tau search.
    for _ in tl.static_range(N_ITER):
        acc_ff = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc_df = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc_ddf = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for i in range(0, n_tiles):
            offs_n = i * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_chunks
            s_tile = tl.load(
                s_row_ptrs + offs_n[None, :] * stride_sn,
                mask=valid_m[:, None] & mask_n[None, :],
                other=NEG_FILL,
            ).to(tl.float32)
            x = s_tile * SCALAR
            xm = ((x > t[:, None]) & mask_n[None, :]).to(tl.float32)
            xa = (x - t[:, None]) * xm
            acc_ff += tl.sum(xa * xa, axis=1)
            acc_df += tl.sum(xa, axis=1)
            acc_ddf += tl.sum(xm, axis=1)
        ff = acc_ff - 1.0
        df = -2.0 * acc_df
        ddf = 2.0 * acc_ddf
        denom = 2.0 * df * df - ff * ddf
        new_t = t - (2.0 * ff * df) / denom
        t_lo = tl.where(ff > 0, t, t_lo)
        t_hi = tl.where(ff < 0, t, t_hi)
        is_good = (new_t > t_lo - EPS) & (new_t < t_hi + EPS)
        t = tl.where(is_good, new_t, 0.5 * (t_lo + t_hi))

    # log_g accumulation pass: only when ESTIMATE_DIAG. Uses the final tau.
    # Computes log_g = mean of log(mean_prob) over cached chunks where mean_prob > EPS_PRIOR.
    # EPS_PRIOR doubles as positivity threshold and log floor, anything smaller is
    # treated as zero so we never log values that would land in the subnormal range.
    EPS_PRIOR: tl.constexpr = 1.0e-20
    sum_log_prior = tl.zeros((), dtype=tl.float32)
    n_pos_total = tl.zeros((), dtype=tl.int32)
    if ESTIMATE_DIAG:
        for i in range(0, n_tiles):
            offs_n = i * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_chunks
            s_tile = tl.load(
                s_row_ptrs + offs_n[None, :] * stride_sn,
                mask=valid_m[:, None] & mask_n[None, :],
                other=NEG_FILL,
            ).to(tl.float32)
            x = s_tile * SCALAR
            xm = ((x > t[:, None]) & mask_n[None, :]).to(tl.float32)
            xa = (x - t[:, None]) * xm
            prob = xa * xa
            prob = tl.where(valid_m[:, None], prob, 0.0)
            mp = tl.sum(prob, axis=0) / GROUPS  # [BLOCK_N]
            pos = (mp > EPS_PRIOR) & mask_n
            lp = tl.where(pos, tl.log(mp), 0.0)
            sum_log_prior += tl.sum(lp, axis=0)
            n_pos_total += tl.sum(pos.to(tl.int32), axis=0)
    safe_n_pos = tl.maximum(n_pos_total, 1).to(tl.float32)
    log_g = sum_log_prior / safe_n_pos
    n_pos_is_zero = n_pos_total == 0

    # Phase B: streaming output, one int32 per BLOCK_N=32 chunk tile.
    p_base = PRIOR + pid_b * stride_pb + pid_hkv * stride_ph
    b_base = BMASK + pid_b * stride_bb + pid_hkv * stride_bh
    bit_pos = tl.arange(0, BLOCK_N).to(tl.int32)

    for j in range(0, n_ints):
        offs_n = j * BLOCK_N + tl.arange(0, BLOCK_N)
        score_mask = offs_n < n_chunks
        diag_pos = offs_n == n_chunks
        prior_mask = offs_n < (n_chunks + 1)

        s_tile = tl.load(
            s_row_ptrs + offs_n[None, :] * stride_sn,
            mask=valid_m[:, None] & score_mask[None, :],
            other=NEG_FILL,
        ).to(tl.float32)
        x = s_tile * SCALAR
        xm = ((x > t[:, None]) & score_mask[None, :]).to(tl.float32)
        xa = (x - t[:, None]) * xm
        prob = xa * xa
        prob = tl.where(valid_m[:, None], prob, 0.0)
        # GQA mean across the `groups` Hq heads for this kv head.
        mean_prob = tl.sum(prob, axis=0) / GROUPS  # [BLOCK_N]

        if ESTIMATE_DIAG:
            pos = (mean_prob > EPS_PRIOR) & score_mask
            log_prior = tl.where(pos, tl.log(mean_prob), 0.0)
            prior_tilde = tl.where(pos, tl.exp((log_prior - log_g) / sigma), 0.0)
            prior_tilde = tl.where(n_pos_is_zero, 0.0, prior_tilde)
            prior_val = tl.where(diag_pos, 1.0, prior_tilde)
        else:
            prior_val = tl.where(diag_pos, 1.0, mean_prob)
        tl.store(p_base + offs_n * stride_pn, prior_val, mask=prior_mask)

        flag = ((mean_prob > 0.0) & score_mask) | diag_pos
        bits = flag.to(tl.int32) << bit_pos
        int_val = tl.sum(bits, axis=0)
        tl.store(b_base + j * stride_bi, int_val)


def block_selection_triton(q, k_blk, chunk_size, attn_mask=None, estimate_diag=False, sigma=1.0e6):
    # q:     [B, Hq,  1, D]
    # k_blk: [B, Hkv, nchunks, D]   nchunks = floor(N / chunk_size)
    # attn_mask: [B, N] bool (True = valid token), optional; used for left-padded seqs
    # returns:
    #   bmask: [B, Hkv, 1, n_ints] int32
    #     n_ints = ceil((nchunks + 1) / 32)
    #     Bit i of int j (LSB = bit 0) flags whether block (j*32 + i) has
    #     positive entmax15 probability after GQA mean. Bit nchunks (the
    #     appended current partial chunk) is always 1. Padding bits are 0.
    #   prior: [B, Hkv, 1, nchunks + 1] fp32
    #     GQA-mean entmax15 probabilities with a trailing 1.0 column.
    B, Hq, _, D = q.shape
    _, Hkv, nchunks, _ = k_blk.shape
    groups = Hq // Hkv
    n_ints = (nchunks + 1 + 31) // 32
    device = q.device

    if nchunks == 0:
        prior = torch.ones(B, Hkv, 1, 1, device=device, dtype=torch.float32)
        bmask = torch.zeros(B, Hkv, 1, n_ints, device=device, dtype=torch.int32)
        bmask[..., 0] = 1
        return bmask, prior

    has_attn_mask = attn_mask is not None
    if has_attn_mask:
        left_pad = attn_mask.shape[-1] - attn_mask.sum(dim=-1)  # [B]
        first_valid_chunk = (left_pad // chunk_size).to(torch.int32).to(device)
    else:
        first_valid_chunk = q  # dummy, not accessed when HAS_ATTN_MASK=False

    BLOCK_M = max(16, triton.next_power_of_2(groups))
    BLOCK_N_K1 = 64 if nchunks >= 64 else max(16, triton.next_power_of_2(nchunks))
    BLOCK_N = 32  # required for inline bit-packing in kernel 2

    scores = torch.empty(B, Hq, nchunks, device=device, dtype=torch.float32)
    max_buf = torch.empty(B, Hq, device=device, dtype=torch.float32)

    _scores_and_max_kernel[(B, Hkv)](
        q, k_blk, scores, max_buf, first_valid_chunk,
        nchunks,
        q.stride(0), q.stride(1), q.stride(3),
        k_blk.stride(0), k_blk.stride(1), k_blk.stride(2), k_blk.stride(3),
        scores.stride(0), scores.stride(1), scores.stride(2),
        max_buf.stride(0), max_buf.stride(1),
        1.0 / math.sqrt(D),
        GROUPS=groups,
        D=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N_K1,
        HAS_ATTN_MASK=has_attn_mask,
    )

    prior = torch.empty(B, Hkv, 1, nchunks + 1, device=device, dtype=torch.float32)
    bmask = torch.empty(B, Hkv, 1, n_ints, device=device, dtype=torch.int32)

    _entmax_mean_pack_kernel[(B, Hkv)](
        scores, max_buf, prior, bmask,
        nchunks,
        scores.stride(0), scores.stride(1), scores.stride(2),
        max_buf.stride(0), max_buf.stride(1),
        prior.stride(0), prior.stride(1), prior.stride(3),
        bmask.stride(0), bmask.stride(1), bmask.stride(3),
        n_ints,
        float(sigma),
        GROUPS=groups,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        N_ITER=3,
        ESTIMATE_DIAG=estimate_diag,
    )

    return bmask, prior

