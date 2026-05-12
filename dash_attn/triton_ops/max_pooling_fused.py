import triton
import triton.language as tl
import torch

MAX_M = 512

@triton.jit
def infllmv2_pool_kernel(
    q_ptr,              # [H, B, N]
    max_seq_len,        # int32
    pooled_q_ptr,       # [H, B, M]
    q_stride_h,         # int32
    q_stride_b,         # int32
    q_stride_n,         # int32
    pooled_q_stride_h,         # int32
    pooled_q_stride_b,         # int32
    pooled_q_stride_m,         # int32
    H,                          # int32
    B,                          # int32
    M,                          # int32
    kernel_size: tl.constexpr,
    pool_stride: tl.constexpr,
    padding: tl.constexpr,
    set_size: tl.constexpr,
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
):
    # grid: (H, B, M)
    tidx_h = tl.program_id(0)  # head
    tidx_b = tl.program_id(1)  # batch idx
    tidx_m = tl.program_id(2)  # window idx

    block_idx = tl.arange(0, 8)
    n_offset = tidx_m * pool_stride - padding
    q_beg_pos = q_ptr + tidx_h * q_stride_h + tidx_b * q_stride_b + n_offset * q_stride_n
    q_block_ptrs = q_beg_pos + block_idx * q_stride_n
    mask = (block_idx + n_offset < max_seq_len) & (block_idx + n_offset >= 0) & (block_idx < set_size)
    q_block_scores = tl.load(
        q_block_ptrs,
        mask=mask,
        other=-float("inf"),
    )
    acc_q = tl.max(q_block_scores, axis=0)

    boundary_mask = (tidx_m < init_blocks) | (M - tidx_m <= local_blocks)
    out_q = tl.where(boundary_mask, float("inf"), acc_q)

    tl.store(
        pooled_q_ptr + tidx_b * pooled_q_stride_b + tidx_h * pooled_q_stride_h + tidx_m * pooled_q_stride_m,
        out_q,
        mask=tidx_m < M,
    )


@triton.jit
def infllmv2_pool_kernel_static(
    q_ptr,              # [H, B, N_max] - pre-allocated buffer
    max_seq_len_ptr,    # pointer to scalar tensor with actual seqlen
    actual_M_ptr,       # pointer to scalar tensor with actual M
    pooled_q_ptr,       # [H, B, M_max] - pre-allocated buffer
    q_stride_h,         # int32
    q_stride_b,         # int32
    q_stride_n,         # int32
    pooled_q_stride_h,         # int32
    pooled_q_stride_b,         # int32
    pooled_q_stride_m,         # int32
    H,                          # int32
    B,                          # int32
    M_max,                      # int32 (buffer size, not actual)
    kernel_size: tl.constexpr,
    pool_stride: tl.constexpr,
    padding: tl.constexpr,
    set_size: tl.constexpr,
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
):
    # grid: (H, B, M_max)
    tidx_h = tl.program_id(0)  # head
    tidx_b = tl.program_id(1)  # batch idx
    tidx_m = tl.program_id(2)  # window idx

    # Load actual sizes from tensor pointers
    max_seq_len = tl.load(max_seq_len_ptr)
    actual_M = tl.load(actual_M_ptr)

    block_idx = tl.arange(0, 8)
    n_offset = tidx_m * pool_stride - padding
    q_beg_pos = q_ptr + tidx_h * q_stride_h + tidx_b * q_stride_b + n_offset * q_stride_n
    q_block_ptrs = q_beg_pos + block_idx * q_stride_n
    mask = (block_idx + n_offset < max_seq_len) & (block_idx + n_offset >= 0) & (block_idx < set_size)
    q_block_scores = tl.load(
        q_block_ptrs,
        mask=mask,
        other=-float("inf"),
    )
    acc_q = tl.max(q_block_scores, axis=0)

    boundary_mask = (tidx_m < init_blocks) | (actual_M - tidx_m <= local_blocks)
    out_q = tl.where(boundary_mask, float("inf"), acc_q)
    
    # For positions >= actual_M, write -inf so topk won't select them
    out_q = tl.where(tidx_m < actual_M, out_q, -float("inf"))

    tl.store(
        pooled_q_ptr + tidx_b * pooled_q_stride_b + tidx_h * pooled_q_stride_h + tidx_m * pooled_q_stride_m,
        out_q,
    )


# 这里假定：1) 必须是decode，2) 格式是齐头pad转unpad，所有batch一样长
def infllmv2_pooling(score, max_seqlen, score_pooled, block_size=32, stride=16, padding=1, set_size=5, init_blocks=1, local_blocks=16):
    # input shape: (H, B, N)
    # pooled shape: (H, B, M)
    H, B = score.shape[:2]
    M = score_pooled.shape[-1]

    grid = (H, B, M)
    infllmv2_pool_kernel[grid](
        score,
        max_seqlen,
        score_pooled,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        score_pooled.stride(0),
        score_pooled.stride(1),
        score_pooled.stride(2),
        H, B, M, block_size, stride, padding, set_size, init_blocks, local_blocks+1,
    )
    return


def infllmv2_pooling_static(score, max_seqlen_tensor, actual_M_tensor, score_pooled, block_size=32, stride=16, padding=1, set_size=5, init_blocks=1, local_blocks=16):
    """
    Static buffer version for CUDA Graph compatibility.
    - score: pre-allocated buffer (H, B, N_max), only [:, :, :max_seqlen] is valid
    - max_seqlen_tensor: scalar tensor with actual seqlen (for input boundary)
    - actual_M_tensor: scalar tensor with actual M (for output boundary)
    - score_pooled: pre-allocated buffer (H, B, M_max)
    """
    H, B = score.shape[:2]
    M_max = score_pooled.shape[-1]

    grid = (H, B, M_max)
    infllmv2_pool_kernel_static[grid](
        score,
        max_seqlen_tensor,
        actual_M_tensor,
        score_pooled,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        score_pooled.stride(0),
        score_pooled.stride(1),
        score_pooled.stride(2),
        H, B, M_max, block_size, stride, padding, set_size, init_blocks, local_blocks+1,
    )
    return


def main():
    import torch
    torch.manual_seed(0)
    device = "cuda"

    H = 4
    B = 2
    N = 1432
    kernel_size = 32
    stride = 16

    score = torch.randn(H, B, N, device=device, dtype=torch.bfloat16)

    M = 88

    score_pooled = torch.empty(H, B, M, device=device, dtype=torch.bfloat16)

    infllmv2_pooling(
        score,
        N,
        score_pooled,
        kernel_size,
        stride,
    )

    baseline = torch.zeros(H, B, M, device=device, dtype=torch.float32)

    for h in range(H):
        for b in range(B):
            seq = score[h, b].unsqueeze(0).unsqueeze(0)     # [1,1,N]
            pooled = torch.nn.functional.max_pool1d(seq.float(), kernel_size, stride)
            baseline[h, b, :pooled.size(-1)] = pooled.squeeze(0).squeeze(0)

    diff = (score_pooled.float() - baseline).abs().max()
    print("Max diff:", diff.item())

if __name__ == "__main__":
    main()
