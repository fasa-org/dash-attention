import torch
import torch.nn.functional as F
import math
import numpy as np
import triton
import triton.language as tl


@triton.jit
def _transform_score_kernel(
    s_ptr,  # score, shape: [num_heads, q_len, k_len]
    bs_ptr,  # block wise score: [num_heads, q_len, num_k_block]
    offs,
    cu_seqlens_q,
    # shape
    num_heads,
    num_offs,
    max_k_len,
    max_blocks,
    pad_len,
    # kernel & block size
    block_size,
    block_stride,  # block_size // kernel_stride
    init_blocks,
    local_blocks,
    # stride
    stride_sh,
    stride_sq,
    stride_sk,
    stride_bsh,
    stride_bsq,
    stride_bsk,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_O: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_b = pid_bh // num_heads
    pid_h = pid_bh % num_heads
    pid_q = tl.program_id(1)
    pid_k = tl.program_id(2)
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    k_start = pid_k * BLOCK_SIZE_K
    if pid_q * BLOCK_SIZE_Q >= q_len:
        return
    # load weight
    off_o = tl.arange(0, BLOCK_SIZE_O)
    w = tl.load(offs + off_o, mask=off_o < num_offs, other=0)
    # load score
    off_q = pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    off_k = (k_start + tl.arange(0, BLOCK_SIZE_K)) * block_stride - pad_len
    off_k = off_k[None, :] + off_o[:, None]
    s_ptrs = (
        s_ptr
        + q_start * stride_sq
        + pid_h * stride_sh
        + off_q[:, None, None] * stride_sq
        + off_k[None, :, :] * stride_sk
    )
    # weighted sum, [BQ, BO, BK] * [1, BO, 1] -> [BQ, BO, BK] -> [BQ, BK]
    s = tl.load(
        s_ptrs,
        mask=(off_q < q_len)[:, None, None] & (off_k >= 0) & (off_k < max_k_len),
        other=0,
    )
    s = s * w[None, :, None]
    s = tl.max(s, axis=1)
    # init mask and local mask
    off_bq = off_q // block_size
    off_bk = k_start + tl.arange(0, BLOCK_SIZE_K)
    s = tl.where(
        (off_bq[:, None] >= off_bk[None, :])  # causal mask
            & (off_bq[:, None] <= off_bk[None, :] + local_blocks),  # local window
        float("inf"),
        s,
    )
    s = tl.where(        
        (off_bk[None, :] < init_blocks),  # init window
        float("inf"),
        s,
    )
    # store block wise score
    bs_ptrs = (
        bs_ptr
        + q_start * stride_bsq
        + pid_h * stride_bsh
        + off_q[:, None] * stride_bsq
        + off_bk[None, :] * stride_bsk
    )
    tl.store(
        bs_ptrs,
        s,
        mask=(off_q < q_len)[:, None] & (off_bk < max_blocks)[None, :],
    )


def transform_score(
    score: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    init_blocks: int = 1,
    local_blocks: int = 2,
) -> torch.Tensor:
    num_k_heads, total_query_len, max_key_len = score.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    pad_len = kernel_size // kernel_stride - 1
    max_blocks = math.ceil(max_seqlen_q / block_size)
    block_score = torch.zeros(
        num_k_heads,
        total_query_len,
        max_blocks,
        dtype=torch.float32,
        device=score.device,
    )
    offs = (
        torch.arange(kernel_size // kernel_stride, device=score.device)[:, None]
        + torch.arange(block_size // kernel_stride, device=score.device)[None, :]
    ).view(-1)
    offs = torch.histc(offs, bins=offs.max() + 1, min=0, max=offs.max())
    num_offs = int(offs.shape[0])
    BLOCK_SIZE_K = min(128, triton.next_power_of_2(max_blocks))
    BLOCK_SIZE_O = triton.next_power_of_2(num_offs)
    BLOCK_SIZE_Q = 8
    grid = (
        num_k_heads * batch_size,
        triton.cdiv(total_query_len, BLOCK_SIZE_Q),
        triton.cdiv(max_blocks, BLOCK_SIZE_K),
    )
    _transform_score_kernel[grid](
        score,
        block_score,
        torch.ones_like(offs, dtype = offs.dtype, device = offs.device),
        cu_seqlens_q,
        num_k_heads,
        offs.shape[0],
        max_key_len,
        max_blocks,
        pad_len,
        block_size,
        block_size // kernel_stride,
        init_blocks,
        local_blocks,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        block_score.stride(0),
        block_score.stride(1),
        block_score.stride(2),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_O=BLOCK_SIZE_O,
        num_warps=8,
        num_stages=3,
    )
    return block_score
