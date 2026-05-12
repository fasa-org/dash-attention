import torch
import triton
import triton.language as tl

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


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

@triton.autotune(
    configs=get_configs(),
    key=["D"],
)
@triton.jit
def _localatt_chunk_kernel(
    keys_ptr, head_cls_ptr, key_chunks_ptr, first_valid_ptr,
    # strides for keys [B, H, N, D]
    stride_kb, stride_kh, stride_kn, stride_kd,
    # strides for head_cls [H, D]
    stride_hh, stride_hd,
    # strides for key_chunks [B, H, nchunks, D]
    stride_ob, stride_oh, stride_oc, stride_od,
    N,
    D: tl.constexpr,
    CHUNK: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    c = tl.program_id(2)

    first_valid = tl.load(first_valid_ptr + b)
    chunk_start = c * CHUNK

    d_offs   = tl.arange(0, D)
    tok_offs = tl.arange(0, CHUNK)
    abs_pos  = chunk_start + tok_offs  # [CHUNK]

    # Load query
    q = tl.load(head_cls_ptr + h * stride_hh + d_offs * stride_hd)

    # Load keys: keys[b, h, abs_pos, :] [CHUNK, D]
    # Positions beyond N are alignment padding; load as 0.0 (not masked to -inf)
    in_seq = abs_pos < N
    k_ptrs = (
        keys_ptr
        + b * stride_kb
        + h * stride_kh
        + abs_pos[:, None] * stride_kn
        + d_offs[None, :] * stride_kd
    )
    k = tl.load(k_ptrs, mask=in_seq[:, None], other=0.0)

    # Attention scores [CHUNK]; mask only left-padding positions
    scale  = 1.0 / (D ** 0.5)
    scores = tl.sum(q[None, :] * k, axis=1) * scale
    real   = abs_pos >= first_valid # this takes care of left padding
    scores = tl.where(real, scores, float('-inf'))

    # Softmax (all-padding chunks produce NaN; handled by stage 2 prefill)
    m     = tl.max(scores, axis=0)
    exp_s = tl.exp(scores - m)
    probs = exp_s / tl.sum(exp_s, axis=0)

    # Weighted sum of keys (= values since K=V) → float32 [D]
    out = tl.sum(probs[:, None] * k, axis=0)

    o_ptrs = (
        key_chunks_ptr
        + b * stride_ob
        + h * stride_oh
        + c * stride_oc
        + d_offs * stride_od
    )
    tl.store(o_ptrs, out)


def localatt_kernel(keys: torch.Tensor, head_cls: torch.Tensor, chunk_size: int, attn_mask: torch.Tensor | None = None):
    assert keys.dim() == 4
    assert head_cls.dim() == 2

    B, H, N, D = keys.shape
    assert head_cls.size(0) == H and head_cls.size(1) == D
    assert D & (D - 1) == 0, f"D must be a power of 2, got {D}"
    assert chunk_size & (chunk_size - 1) == 0, f"chunk_size must be a power of 2, got {chunk_size}"

    nchunks = _ceil_div(N, chunk_size)

    if attn_mask is not None:
        first_valid = (N - attn_mask.sum(dim=-1)).to(torch.int32)
    else:
        first_valid = keys.new_zeros(B, dtype=torch.int32)

    # the last nchunk will not be used.
    key_chunks = torch.empty(B, H, nchunks, D, dtype=keys.dtype, device=keys.device)

    _localatt_chunk_kernel[(B, H, nchunks)](
        keys, head_cls, key_chunks, first_valid,
        *keys.stride(), *head_cls.stride(), *key_chunks.stride(),
        N=N,
        D=D, CHUNK=chunk_size,
    )

    return key_chunks
