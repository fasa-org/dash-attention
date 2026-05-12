import pytest
import torch

from dash_attn import dash_attn
from dash_attn.dash_attn_interface import _count_active_blocks


def test_count_active_blocks_bitpacked_mask():
    bmask = torch.tensor([[[[0], [1], [3], [-1], [-2147483648]]]], dtype=torch.int32)
    expected = torch.tensor([[[0, 1, 2, 32, 1]]], dtype=torch.int32)
    assert torch.equal(_count_active_blocks(bmask), expected)


def test_dash_attn_prefill_smoke():
    torch.manual_seed(0)

    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    batch = 1
    kv_heads = 2
    query_heads = 16
    seq_len = 1024
    head_dim = 128
    chunk_size = 64

    queries = torch.randn(batch, query_heads, seq_len, head_dim, device=device, dtype=dtype).contiguous()
    keys = torch.randn(batch, kv_heads, seq_len, head_dim, device=device, dtype=dtype).contiguous()
    values = torch.randn(batch, kv_heads, seq_len, head_dim, device=device, dtype=dtype).contiguous()
    head_cls = torch.randn(kv_heads, head_dim, device=device, dtype=dtype).contiguous()

    model = dash_attn(
        chunk_size=chunk_size,
        enable_gqa=True,
        estimate_diagonal=True,
        return_active_blocks=True,
    )

    out, active_blocks = model(queries, keys, values, head_cls)
    torch.cuda.synchronize()

    assert out.shape == queries.shape
    assert active_blocks.shape == (batch, kv_heads, seq_len)
    assert active_blocks.dtype == torch.int32
    assert torch.isfinite(out).all()

if __name__ == "__main__":
    test_dash_attn_prefill_smoke()