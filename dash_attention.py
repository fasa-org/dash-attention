# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from functools import lru_cache
import torch
from torch import nn

# torch.autograd.set_detect_anomaly(True)

from megatron.core import packed_seq_params, mpu
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.utils import print_rank_0

try:
    from infllm_v2 import (
        infllmv2_attn_stage1,
        infllmv2_attn_varlen_func,
        max_pooling_1d_varlen
    )
    from .infllmv2_entmax import compressed_attention_fwd
    from .infllmv2_entmax.transform_score import transform_score
    from adasplash import triton_entmax
    from flash_attn import flash_attn_func
except:
    pass

from .infllmv2_entmax import topk_sparse_attention
import torch.nn.functional as F


@lru_cache(maxsize=16)
def calc_chunks_with_stride(cu_seqlen, moba_chunk_size, kernel_stride):

    batch_sizes = cu_seqlen[1:] - cu_seqlen[:-1]


    max_seq_len = torch.max(batch_sizes)
    max_num_chunks_per_seq = (max_seq_len - moba_chunk_size) // kernel_stride + 1 
    chunk_start_offsets = torch.arange(0, max_num_chunks_per_seq * kernel_stride, kernel_stride, device=cu_seqlen.device)
    seq_starts = cu_seqlen[:-1]
    chunk_start_in_seq = seq_starts[:, None] + chunk_start_offsets[None, :] 


    chunk_end_in_seq = chunk_start_in_seq + moba_chunk_size
    valid_chunk_mask = (chunk_end_in_seq <= (seq_starts[:, None] + batch_sizes[:, None]))  # 完整 chunk


    valid_chunk_starts = chunk_start_in_seq[valid_chunk_mask]  # [num_valid_chunks]
    del chunk_start_in_seq

    chunk_indices = torch.arange(
        0, moba_chunk_size, device=cu_seqlen.device
    )[None, :]  # [1, moba_chunk_size]
    filtered_indices = valid_chunk_starts[:, None] + chunk_indices  # [num_valid_chunks, moba_chunk_size]
    filtered_indices = filtered_indices.view(-1)  


    num_filtered_chunks_per_batch = valid_chunk_mask.sum(dim=1)
    cu_seqlens_compressed = torch.zeros(
        len(cu_seqlen), dtype=torch.int32, device=cu_seqlen.device
    )
    cu_seqlens_compressed[1:] = num_filtered_chunks_per_batch.cumsum(dim=0)
    # del num_filtered_chunks_per_batch, chunk_start_offsets, seq_starts, chunk_end_in_seq, valid_chunk_mask, chunk_indices, valid_chunk_starts
    # torch.cuda.empty_cache()
    return filtered_indices, cu_seqlens_compressed


def compressed_attention(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    topk: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float = None,
    init_blocks: int = 1,
    local_blocks: int = 2,
    cache_lens=None,
) -> torch.Tensor:
    """Calculate topk indices using infllmv2_stage1 and max_pooling."""
    # with torch.no_grad():
    batch_size = cu_seqlens_q.shape[0] - 1
    
    # Always prefilling stage
    # Calculate q_idx for each query position in each batch
    cache_lens = torch.zeros(batch_size, dtype=torch.int32, device=q.device) 
    q_idx = torch.cat([
        (torch.arange(cu_seqlens_q[i + 1] - cu_seqlens_q[i], device=q.device) + 
            max_seqlen_q - (cu_seqlens_q[i + 1] - cu_seqlens_q[i])) // block_size
        for i in range(batch_size)
    ], dim=0)  # shape: [total_q_len]

    # Calculate attention score
    score = compressed_attention_fwd(
        q, k, kernel_size, kernel_stride, 
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        q.shape[-1] ** (-0.5)
    )

    # NOTE(Yuxiang Huang): do entmax per query head
    score = triton_entmax(score, alpha=1.5, n_iter=3)

    score_ = score[:, :q_idx.shape[0], :] 

    score = score.reshape(k.shape[1], -1, score.shape[-2], score.shape[-1])
    # NOTE(Yuxiang Huang): Then, head aggregation by mean pooling
    score = score.mean(dim=1)
    score = score[:, :q_idx.shape[0], :]  # [num_heads, total_q_len, num_blocks]

    pooled_score = score_
    pooled_score = F.pad(pooled_score, (0, 1), value=0) # This is is to simulate `transform_score` to avoid illegal mem access

    block_score = transform_score(
        score.contiguous(),
        kernel_size,
        kernel_stride,
        block_size,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
    )

    bmask = block_score > 0
    max_k = bmask.sum(dim=-1).max().item()
    topk_idx = block_score.topk(k=max_k, dim=-1).indices
    bmask_indexed = torch.gather(bmask, dim=-1, index=topk_idx)
    topk_idx[~bmask_indexed] = -1
    topk_idx[topk_idx > q_idx[None, :, None]] = -1
    topk_idx = topk_idx.to(torch.int32)
        
    return topk_idx, max_k, pooled_score


class CompressK(torch.nn.Module):
    def __init__(self, head_num_k, head_dim, kernel_size, kernel_stride=16):
        super().__init__()
        self.kernel_size = kernel_size
        self.head_num_k = head_num_k
        self.head_dim = head_dim
        self.kernel_stride = kernel_stride

    def forward(self, indexer_q: torch.Tensor, k: torch.Tensor, cu_seqlens):
        filtered_k_indices, cu_seqlens_compressed = calc_chunks_with_stride(
            cu_seqlens, self.kernel_size, self.kernel_stride
        )

        filtered_k = k.index_select(0, filtered_k_indices.view(-1))
        filtered_q = indexer_q.index_select(0, filtered_k_indices.view(-1))

        filtered_k = filtered_k.view(filtered_k.shape[0] // self.kernel_size, self.kernel_size, self.head_num_k, self.head_dim)  #[l, block_size, h, d]
        filtered_q = filtered_q.view(filtered_q.shape[0] // self.kernel_size, self.kernel_size, self.head_num_k, self.head_dim)  #[l, block_size, h, d]

        # NOTE(Yuxiang Huang): use mean query as the input
        filtered_q = filtered_q.mean(dim=1)

        # local attention [FA] NOTE(Yuxiang Huang): this is slower than einsum
        # filtered_q = filtered_q.unsqueeze(1)  
        # compress_k = flash_attn_func(filtered_q, filtered_k, filtered_k, dropout_p=0, softmax_scale=None, causal=False)
        # compress_k = compress_k.squeeze(1)

        # local attention
        comp_score = torch.einsum('nhd,nmhd->nmh', filtered_q, filtered_k)
        comp_prob = torch.nn.functional.softmax(comp_score * (filtered_k.shape[-1] ** -0.5), dim=1)
        compress_k = torch.einsum('nmh,nmhd->nhd', comp_prob, filtered_k)


        return compress_k, cu_seqlens_compressed


class DashAttnDotProductAttention(DotProductAttention):
    """Native Sparse Attention implementation."""

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        window_size: int = 0,
        kernel_size: int = 64,
        kernel_stride: int = 64,
        block_size: int = 64,
        topk = 32,
        init_blocks: int = 0,
        local_blocks: int = 0,
        **kwargs
    ):
        # Initialize parent without NSA params
        super().__init__(
            config=config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            **kwargs
        )
        
        # Set NSA specific parameters
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride
        self.block_size = block_size
        self.init_blocks = init_blocks
        self.local_blocks = self.window_size // self.block_size
        self.topk = topk
        
        # Print parameters
        print_rank_0(f'self.topk: {self.topk}')
        print_rank_0(f'[NSA Init] num_attention_heads_per_partition: {self.num_attention_heads_per_partition}')
        print_rank_0(f'[NSA Init] num_query_groups_per_partition: {self.num_query_groups_per_partition}')
        print_rank_0(f'[NSA Init] hidden_size_per_attention_head: {self.hidden_size_per_attention_head}')
        print_rank_0(f'[NSA Init] hidden_size_per_partition: {self.hidden_size_per_partition}')
        
        self.compress_k = CompressK(
            self.num_query_groups_per_partition, 
            self.hidden_size_per_attention_head,
            kernel_size=self.kernel_size,
            kernel_stride=self.kernel_stride
        )
        
        self.dropout_p = config.attention_dropout if ('attention_dropout' not in kwargs or kwargs['attention_dropout'] is None) else kwargs['attention_dropout']
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        attn_mask_type=AttnMaskType.padding,
        query_lengths=None,
        key_lengths=None,
        packed_seq_params=None,
        indexer_q=None,
        query_nope=None,
        key_nope=None,
    ):
        assert packed_seq_params is not None, "cu_seqlens needed!"
        
        # # Log input shapes
        # print_rank_0(f"[NSA] Input shapes - query: {query_layer.shape}, key: {key_layer.shape}, value: {value_layer.shape}, packed_seq_params.cu_seqlens_q: {packed_seq_params.cu_seqlens_q}")
        
        if packed_seq_params.max_seqlen_q == None:
            seqlens_q = packed_seq_params.cu_seqlens_q[1:] - packed_seq_params.cu_seqlens_q[:-1]
            max_seqlen_q = seqlens_q.max().item()
        else:
            max_seqlen_q = packed_seq_params.max_seqlen_q
            
        if packed_seq_params.max_seqlen_kv == None:
            seqlens_kv = packed_seq_params.cu_seqlens_kv[1:] - packed_seq_params.cu_seqlens_kv[:-1]
            max_seqlen_kv = seqlens_kv.max().item()
        else:
            max_seqlen_kv = packed_seq_params.max_seqlen_kv

        # Always run compressed attention, use NoPE for local attn
        compressed_k, compressed_cu_seqlens = self.compress_k(indexer_q, key_layer, packed_seq_params.cu_seqlens_q)
        compressed_seqlens = compressed_cu_seqlens[1:] - compressed_cu_seqlens[:-1]


        topk_idx, max_k, pooled_score = compressed_attention(
            self,
            query_layer,
            # query_nope,
            # query_layer.detach(),
            compressed_k,
            compressed_k, # placeholder
            self.kernel_size,
            self.kernel_stride,
            self.block_size,
            self.topk,
            packed_seq_params.cu_seqlens_q,
            compressed_cu_seqlens,
            max_seqlen_q,
            compressed_seqlens.max().item(),
            None,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
        )

        topk_attn_output = None

        # NOTE(Yuxiang Huang): map entmax scores back to logit space (R^n) NOTE on 03.14: The following code provides a better precision
        mask = pooled_score > 0
        mask_cnt = mask.sum(dim=-1, keepdim=True)
        ps_valid = torch.zeros_like(pooled_score)
        ps_valid[mask] = torch.log(pooled_score[mask] / self.block_size)
        ps_mean = ps_valid.sum(dim=-1, keepdim=True) / mask_cnt.clamp(min=1)

        # NOTE: choose one from the two implementations blow
        # NOTE(Yuxiang Huang): sigma = 1 is not good enough, even sigma=100 causes large performance drop. However, sigma=1e6 works fine.
        sigma = 1e6
        pooled_score = (ps_valid - ps_mean) / sigma
        
        # NOTE(Yuxiang Huang): straight through estimator
        # pooled_score = pooled_score - ps_valid.detach()


        # triton implementation
        # pooled_score: [H, N, max_seqlen_k]
        topk_attn_output = topk_sparse_attention(query_layer, key_layer, value_layer, topk_idx, pooled_score, block_size=self.block_size, cu_seqlens=packed_seq_params.cu_seqlens_q)

        return topk_attn_output