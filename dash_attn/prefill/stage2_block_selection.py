import math
import torch
import triton
import triton.language as tl


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------
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
    key=["GROUPS"],
)
@triton.jit
def _entmax_prefill_kernel(
    SCORES,            # [B, Hq,  N, nchunks] fp32
    PRIOR,             # [B, Hkv, N, nchunks] fp32  (output)
    BMASK,             # [B, Hkv, N, n_ints]  i32   (output)
    FIRST_VALID_CHUNK, # [B] i32
    ATTN_MASK,         # [B, N] bool  (only touched when HAS_ATTN_MASK)
    Hkv,
    nchunks,
    n_ints,
    chunk_size,
    stride_sb, stride_sh, stride_sn, stride_sc,
    stride_pb, stride_ph, stride_pn, stride_pc,
    stride_bb, stride_bh, stride_bn, stride_bi,
    stride_fvc,
    stride_amb, stride_amn,
    sigma,
    GROUPS:        tl.constexpr,
    BLOCK_M:       tl.constexpr,
    BLOCK_N:       tl.constexpr,   # must be 32 for inline bit-packing
    N_ITER:        tl.constexpr,
    HAS_ATTN_MASK: tl.constexpr,
    ESTIMATE_DIAG: tl.constexpr,
):
    pid_n   = tl.program_id(0)   # 0 .. N-1
    pid_b   = tl.program_id(1)   # 0 .. B-1
    pid_hkv = tl.program_id(2)   # 0 .. Hkv-1

    EPS: tl.constexpr      = 1.0e-5
    SCALAR: tl.constexpr   = 0.5
    NEG_FILL: tl.constexpr = -1.0e4

    QUERY_CHUNK  = pid_n // chunk_size
    first_valid  = tl.load(FIRST_VALID_CHUNK + pid_b * stride_fvc)

    if HAS_ATTN_MASK:
        is_valid = tl.load(ATTN_MASK + pid_b * stride_amb + pid_n * stride_amn)
    else:
        is_valid = True

    # GQA head indices for this kv head
    offs_m  = tl.arange(0, BLOCK_M)
    valid_m = offs_m < GROUPS
    hq_offs = pid_hkv * GROUPS + offs_m   # [BLOCK_M]

    # Base pointers into SCORES for this (b, n) pair
    s_base     = SCORES + pid_b * stride_sb + pid_n * stride_sn
    s_row_ptrs = s_base + hq_offs[:, None] * stride_sh   # [BLOCK_M, 1]

    # Number of BLOCK_N-sized tiles that may contain causally valid chunks
    n_tiles_causal = tl.cdiv(QUERY_CHUNK, BLOCK_N)

    # ------------------------------------------------------------------
    # Phase 0: find per-head max over chunks [first_valid, QUERY_CHUNK)
    # ------------------------------------------------------------------
    max_val = tl.full((BLOCK_M,), NEG_FILL, dtype=tl.float32)
    for i in range(0, n_tiles_causal):
        offs_c = i * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_c = (offs_c >= first_valid) & (offs_c < QUERY_CHUNK)
        s_tile = tl.load(
            s_row_ptrs + offs_c[None, :] * stride_sc,
            mask=valid_m[:, None] & mask_c[None, :],
            other=NEG_FILL,
        ).to(tl.float32) * SCALAR
        s_for_max = tl.where(valid_m[:, None] & mask_c[None, :], s_tile, NEG_FILL)
        max_val   = tl.maximum(max_val, tl.max(s_for_max, axis=1))

    # Tau initialisation (max already scaled by SCALAR above)
    t_hi = max_val
    t_lo = max_val - 1.0
    t    = 0.5 * (t_lo + t_hi)

    # ------------------------------------------------------------------
    # Phase A: Halley tau search
    # ------------------------------------------------------------------
    for _ in tl.static_range(N_ITER):
        acc_ff  = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc_df  = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc_ddf = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for i in range(0, n_tiles_causal):
            offs_c = i * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_c = (offs_c >= first_valid) & (offs_c < QUERY_CHUNK)
            s_tile = tl.load(
                s_row_ptrs + offs_c[None, :] * stride_sc,
                mask=valid_m[:, None] & mask_c[None, :],
                other=NEG_FILL,
            ).to(tl.float32)
            x      = s_tile * SCALAR
            active = (x > t[:, None]) & mask_c[None, :] & valid_m[:, None]
            xm     = active.to(tl.float32)
            xa     = (x - t[:, None]) * xm
            acc_ff  += tl.sum(xa * xa, axis=1)
            acc_df  += tl.sum(xa,      axis=1)
            acc_ddf += tl.sum(xm,      axis=1)
        ff     = acc_ff - 1.0
        df     = -2.0 * acc_df
        ddf    =  2.0 * acc_ddf
        denom  = 2.0 * df * df - ff * ddf
        new_t  = t - (2.0 * ff * df) / denom
        t_lo   = tl.where(ff > 0, t, t_lo)
        t_hi   = tl.where(ff < 0, t, t_hi)
        is_good = (new_t > t_lo - EPS) & (new_t < t_hi + EPS)
        t       = tl.where(is_good, new_t, 0.5 * (t_lo + t_hi))

    # ------------------------------------------------------------------
    # log_g accumulation pass (only when ESTIMATE_DIAG). Uses final tau.
    # Computes log_g = mean of log(mean_prob) over causally valid chunks where
    # mean_prob > EPS_PRIOR. EPS_PRIOR doubles as positivity threshold and
    # log floor, anything smaller is treated as zero so we never log values
    # that would land in the subnormal range.
    # ------------------------------------------------------------------
    EPS_PRIOR: tl.constexpr = 1.0e-20
    sum_log_prior = tl.zeros((), dtype=tl.float32)
    n_pos_total = tl.zeros((), dtype=tl.int32)
    if ESTIMATE_DIAG:
        for i in range(0, n_tiles_causal):
            offs_c = i * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_c = (offs_c >= first_valid) & (offs_c < QUERY_CHUNK)
            s_tile = tl.load(
                s_row_ptrs + offs_c[None, :] * stride_sc,
                mask=valid_m[:, None] & mask_c[None, :],
                other=NEG_FILL,
            ).to(tl.float32)
            x      = s_tile * SCALAR
            active = (x > t[:, None]) & mask_c[None, :] & valid_m[:, None]
            xm     = active.to(tl.float32)
            xa     = (x - t[:, None]) * xm
            prob   = xa * xa
            prob   = tl.where(valid_m[:, None], prob, 0.0)
            mp     = tl.sum(prob, axis=0) / GROUPS   # [BLOCK_N]
            pos    = (mp > EPS_PRIOR) & mask_c
            lp     = tl.where(pos, tl.log(mp), 0.0)
            sum_log_prior += tl.sum(lp, axis=0)
            n_pos_total   += tl.sum(pos.to(tl.int32), axis=0)
    safe_n_pos    = tl.maximum(n_pos_total, 1).to(tl.float32)
    log_g         = sum_log_prior / safe_n_pos
    n_pos_is_zero = n_pos_total == 0

    # ------------------------------------------------------------------
    # Phase B: compute probs, GQA mean, write prior + packed bmask
    # ------------------------------------------------------------------
    p_base  = PRIOR + pid_b * stride_pb + pid_hkv * stride_ph + pid_n * stride_pn
    b_base  = BMASK + pid_b * stride_bb + pid_hkv * stride_bh + pid_n * stride_bn
    bit_pos = tl.arange(0, BLOCK_N).to(tl.int32)
    valid_f = is_valid.to(tl.float32)   # 1.0 for valid tokens, 0.0 for padding

    for j in range(0, n_ints):
        offs_c     = j * BLOCK_N + tl.arange(0, BLOCK_N)
        score_mask = (offs_c >= first_valid) & (offs_c < QUERY_CHUNK)  # valid entmax range
        diag_pos   = offs_c == QUERY_CHUNK                              # always-attend diagonal
        chunk_mask = offs_c < nchunks                                   # in-bounds guard

        s_tile = tl.load(
            s_row_ptrs + offs_c[None, :] * stride_sc,
            mask=valid_m[:, None] & score_mask[None, :],
            other=NEG_FILL,
        ).to(tl.float32)
        x      = s_tile * SCALAR
        active = (x > t[:, None]) & score_mask[None, :] & valid_m[:, None]
        xm     = active.to(tl.float32)
        xa     = (x - t[:, None]) * xm
        prob   = xa * xa
        prob   = tl.where(valid_m[:, None], prob, 0.0)
        mean_prob = tl.sum(prob, axis=0) / GROUPS   # [BLOCK_N]

        # prior: diagonal = 1.0, previous valid chunks = entmax prob (or
        # flattened prior_tilde when ESTIMATE_DIAG), rest = 0.
        # Zero everything for padding query positions (valid_f = 0).
        if ESTIMATE_DIAG:
            pos = (mean_prob > EPS_PRIOR) & score_mask
            log_prior = tl.where(pos, tl.log(mean_prob), 0.0)
            prior_tilde = tl.where(pos, tl.exp((log_prior - log_g) / sigma), 0.0)
            prior_tilde = tl.where(n_pos_is_zero, 0.0, prior_tilde)
            prior_val = tl.where(diag_pos, valid_f, prior_tilde * valid_f)
        else:
            prior_val = tl.where(diag_pos, valid_f, mean_prob * valid_f)
        tl.store(p_base + offs_c * stride_pc, prior_val, mask=chunk_mask)

        # bmask: active chunks + diagonal, zeroed for padding queries.
        flag    = ((mean_prob > 0.0) & score_mask) | diag_pos
        int_val = tl.sum(flag.to(tl.int32) << bit_pos, axis=0)
        int_val = tl.where(is_valid, int_val, 0)
        tl.store(b_base + j * stride_bi, int_val)


def block_selection(queries, key_chunks, chunk_size, attn_mask=None, estimate_diag=False, sigma=1.0e6):
    """
    queries   : [B, Hq, N, D]
    key_chunks: [B, Hkv, nchunks, D]  nchunks = ceil_div(N, chunk_size)
    attn_mask : [B, N] bool (True = valid), optional

    Returns
    -------
    bmask : [B, Hkv, N, n_ints] int32   n_ints = ceil(nchunks / 32)
    prior : [B, Hkv, N, nchunks] fp32
    """
    B, Hq, N, D   = queries.shape
    _, Hkv, nchunks, _ = key_chunks.shape
    groups = Hq // Hkv
    n_ints = (nchunks + 31) // 32
    device = queries.device

    # Expand keys for GQA then compute scores in fp32
    key_exp = repeat_kv(key_chunks, groups)   # [B, Hq, nchunks, D]
    scale   = 1.0 / math.sqrt(D)
    scores  = torch.matmul(queries.float(), key_exp.float().transpose(-1, -2)) * scale
    scores  = scores.contiguous()             # [B, Hq, N, nchunks]

    if attn_mask is not None:
        left_pad          = attn_mask.shape[-1] - attn_mask.sum(dim=-1)   # [B]
        first_valid_chunk = (left_pad // chunk_size).to(torch.int32).to(device)
    else:
        first_valid_chunk = torch.zeros(B, device=device, dtype=torch.int32)

    prior = torch.empty(B, Hkv, N, nchunks, device=device, dtype=torch.float32)
    bmask = torch.empty(B, Hkv, N, n_ints,  device=device, dtype=torch.int32)

    BLOCK_M = max(16, triton.next_power_of_2(groups))
    BLOCK_N = 32   # fixed for bit-packing

    # When attn_mask is None pass scores as a harmless dummy pointer; the
    # kernel never dereferences it (HAS_ATTN_MASK=False, constexpr).
    _attn  = attn_mask if attn_mask is not None else scores
    _sa_b  = attn_mask.stride(0) if attn_mask is not None else 0
    _sa_n  = attn_mask.stride(1) if attn_mask is not None else 0

    _entmax_prefill_kernel[(N, B, Hkv)](
        scores, prior, bmask, first_valid_chunk, _attn,
        Hkv, nchunks, n_ints, chunk_size,
        scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3),
        prior.stride(0),  prior.stride(1),  prior.stride(2),  prior.stride(3),
        bmask.stride(0),  bmask.stride(1),  bmask.stride(2),  bmask.stride(3),
        first_valid_chunk.stride(0),
        _sa_b, _sa_n,
        float(sigma),
        GROUPS=groups,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        N_ITER=3,
        HAS_ATTN_MASK=attn_mask is not None,
        ESTIMATE_DIAG=estimate_diag,
    )

    return bmask, prior

