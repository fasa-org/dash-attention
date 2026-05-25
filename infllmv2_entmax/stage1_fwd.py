import math
from typing import Any, Tuple, Union
from collections import Counter
import torch
import triton
import triton.language as tl
import warnings
from native_sparse_attention.ops.triton.utils import get_num_warps_stages, is_hopper_gpu


IS_HOPPER_GPU = is_hopper_gpu()


@triton.jit
def forward_kernel(
    q_ptr,  # Q: n x h x d
    k_ptr,  # K: n x h x d
    attn_score_ptr, # S: n x h x d
    # size and stride at compresstion
    kernel_size,
    kernel_stride,
    # seqlens
    cu_seqlens_q,
    cu_seqlens_k,
    # shape
    NUM_KV_HEADS,
    NUM_SHARE_Q_HEADS,
    HEAD_DIM,
    # sm_scale
    sm_scale,
    # stride
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_sh,
    stride_sq,
    stride_sk,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_D: tl.constexpr,
):
    # do not need log2(e) here
    # qk_scale = sm_scale * 1.44269504
    qk_scale = sm_scale
    # get batch id and head id
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(2)
    pid_kh = pid_h // NUM_SHARE_Q_HEADS
    # get q k start and len after rmpad
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    k_start = tl.load(cu_seqlens_k + pid_b)
    k_len = tl.load(cu_seqlens_k + pid_b + 1) - k_start
    # skip first kernel_size query block, because they do no attend to any keys
    q_start_in_seq = pid_q * BLOCK_SIZE_Q + kernel_size - 1
    if q_start_in_seq >= q_len:
        return
    # init qkv pointer
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
        shape=(q_len, HEAD_DIM),
        strides=(stride_qn, stride_qd),
        offsets=(q_start_in_seq, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_D),
        order=(1, 0),
    )
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + k_start * stride_kn + pid_kh * stride_kh,
        shape=(HEAD_DIM, k_len),
        strides=(stride_kd, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K),
        order=(0, 1),
    )
    s_ptrs = tl.make_block_ptr(
        base=attn_score_ptr + pid_h * stride_sh + q_start * stride_sq + 0 * stride_sk,
        shape=(q_len, k_len),
        strides=(stride_sq, stride_sk),
        offsets=(q_start_in_seq, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_K),
        order=(1, 0),
    )
    # load q
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
    # init statistics
    off_q = tl.arange(0, BLOCK_SIZE_Q) + q_start_in_seq
    off_k = tl.arange(0, BLOCK_SIZE_K) * kernel_stride + kernel_size - 1
    # attention
    lo = 0
    hi = min(k_len, (q_start_in_seq + BLOCK_SIZE_Q - kernel_size) // kernel_stride + 1)
    for i in range(lo, hi, BLOCK_SIZE_K):
        i = tl.multiple_of(i, BLOCK_SIZE_K)
        # load k
        k = tl.load(k_ptrs, boundary_check=(1, 0), padding_option="zero")
        # compute qk
        qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
        qk += tl.where(
            off_q[:, None] >= (i * kernel_stride + off_k)[None, :], 0, float("-inf")
        )
        qk += tl.dot(q, k) * qk_scale
        # store s
        tl.store(s_ptrs, qk.to(tl.bfloat16), boundary_check=(0, 1))
        # update ptrs
        k_ptrs = tl.advance(k_ptrs, (0, BLOCK_SIZE_K))
        s_ptrs = tl.advance(s_ptrs, (0, BLOCK_SIZE_K))


def compressed_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
):
    # dtype check
    assert k.dtype == q.dtype 
    assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32
    # shape
    q_len, num_q_heads, head_dim = q.shape
    k_len, num_k_heads, head_dim = k.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    assert q_len > k_len
    # gqa
    assert num_q_heads % num_k_heads == 0
    num_share_q_heads = num_q_heads // num_k_heads
    # output tensor
    attn_score = torch.full((num_q_heads, q_len, max_seqlen_k), float('-inf'), dtype=q.dtype, device=q.device)
    # launch kernel
    grid = lambda META: (
        batch_size,
        num_q_heads,
        triton.cdiv(max_seqlen_q, META["BLOCK_SIZE_Q"]),
    )
    BLOCK_SIZE_Q = 128
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    num_warps, num_stages = get_num_warps_stages(head_dim, BLOCK_SIZE_Q, IS_HOPPER_GPU)
    forward_kernel[grid](
        q,
        k,
        attn_score,
        kernel_size,
        kernel_stride,
        cu_seqlens_q,
        cu_seqlens_k,
        num_k_heads,
        num_share_q_heads,
        head_dim,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        attn_score.stride(0),
        attn_score.stride(1),
        attn_score.stride(2),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return attn_score

