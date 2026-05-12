import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

def localatt_kernel(keys: torch.Tensor, head_cls: torch.Tensor):
    # keys = [B, H, chunk_size, D]
    # head_cls = [H, D]
    # First broadcast head_cls to [B, H, 1, D].
    B, H, _, D = keys.shape
    head_cls = head_cls[None, :, None, :].expand((B, -1, 1, -1))

    # Then do attention sdpa(query=head_cls, keys=keys, values=keys) -> [B, H, 1, D]
    with sdpa_kernel(
        [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION],
        set_priority=True,
    ):
        return F.scaled_dot_product_attention(head_cls, keys, keys, is_causal=False)

# import torch

# try:
#     from ..prefill.stage1_chunk_summarize import localatt_kernel as _prefill_localatt_kernel
# except ImportError:
#     from prefill.stage1_chunk_summarize import localatt_kernel as _prefill_localatt_kernel


# def localatt_kernel(keys: torch.Tensor, head_cls: torch.Tensor):
#     # keys = [B, H, chunk_size, D], head_cls = [H, D]
#     # Reuse the prefill Triton kernel with N=chunk_size so a chunk summarized
#     # during decoding is bit-identical to the same chunk summarized during
#     # prefill. Returns [B, H, 1, D].
#     chunk_size = keys.size(-2)
#     return _prefill_localatt_kernel(keys, head_cls, chunk_size=chunk_size, attn_mask=None)
