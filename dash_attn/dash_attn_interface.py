import torch

from .cache_utils import ChunkRepresentationCache
from . import decoding
from . import prefill

torch.set_float32_matmul_precision("high")


def _count_active_blocks(bmask: torch.Tensor) -> torch.Tensor:
    """Count set bits in a bit-packed block mask of shape [B, H, S, n_ints]."""
    x = bmask.to(torch.int64) & 0xFFFFFFFF
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = ((x + (x >> 4)) & 0x0F0F0F0F) * 0x01010101
    counts = (x & 0xFFFFFFFF) >> 24
    return counts.sum(dim=-1).to(torch.int32)


class dash_attn(torch.nn.Module):
    def __init__(
        self,
        chunk_size: int,
        enable_gqa: bool,
        estimate_diagonal: bool,
        scaling_factor: float = 1.0,
        return_active_blocks: bool = False,
        max_chunks: int = 512,
        sigma: float = 1.0e6,
    ):
        super().__init__()

        self.chunk_size = chunk_size
        self.enable_gqa = enable_gqa
        self.estimate_diagonal = estimate_diagonal
        self.scaling_factor = scaling_factor
        self.return_active_blocks = return_active_blocks
        self.sigma = sigma

        self.chunk_cache = ChunkRepresentationCache(max_chunks=max_chunks)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        head_cls: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        user_bmask: torch.Tensor | None = None,
    ):
        is_decoding = queries.size(-2) == 1 and values.size(-2) > 1
        if is_decoding:
            return self.decode(queries, keys, values, head_cls, attn_mask, user_bmask)
        return self.prefill(queries, keys, values, head_cls, attn_mask, user_bmask)

    def prefill(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        head_cls: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        user_bmask: torch.Tensor | None = None,
    ):
        assert queries.size(-2) == keys.size(-2)
        B, Hkv, _, D = keys.shape
        self.chunk_cache.allocate(B, Hkv, D, keys.dtype, keys.device)
        prefill_result = self._prefill_compiled(queries, keys, values, head_cls, attn_mask, user_bmask)
        if self.return_active_blocks:
            out, key_chunks, active_blocks = prefill_result
        else:
            out, key_chunks = prefill_result
        # Persist all full chunks; drop the trailing chunk, which is always
        # treated as the "current" (possibly partial) chunk and rebuilt by decode.
        self.chunk_cache.append_many(key_chunks[..., :-1, :])
        if self.return_active_blocks:
            return out, active_blocks
        return out

    def decode(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        head_cls: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        user_bmask: torch.Tensor | None = None,
    ):
        if keys.size(-2) % self.chunk_size == 1:
            last_chunk = decoding.summarize_chunk(keys[..., -self.chunk_size - 1 : -1, :], head_cls)
            self.chunk_cache.append(last_chunk)
        keys_chunks = self.chunk_cache.view()
        return self._decode_compiled(queries, keys, values, keys_chunks, attn_mask, user_bmask)

    def _prefill_compiled(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        head_cls: torch.Tensor,
        attn_mask: torch.Tensor | None,
        user_bmask: torch.Tensor | None,
    ):
        key_chunks = prefill.summarize_chunk(keys, head_cls, self.chunk_size, attn_mask)
        bmask, prior = prefill.score_blocks(
            queries * self.scaling_factor,
            key_chunks,
            self.chunk_size,
            attn_mask,
            estimate_diag=self.estimate_diagonal,
            sigma=self.sigma,
        )
        if user_bmask is not None:
            bmask = bmask & user_bmask
        out = prefill.full_attn(queries, keys, values, bmask, self.chunk_size, attn_mask, prior)
        if self.return_active_blocks:
            return out, key_chunks, _count_active_blocks(bmask)
        return out, key_chunks

    def _decode_compiled(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        keys_chunks: torch.Tensor,
        attn_mask: torch.Tensor | None,
        user_bmask: torch.Tensor | None,
    ):
        bmask, prior = decoding.score_blocks(
            queries * self.scaling_factor,
            keys_chunks,
            self.chunk_size,
            attn_mask=attn_mask,
            estimate_diag=self.estimate_diagonal,
            sigma=self.sigma,
        )
        if user_bmask is not None:
            bmask = user_bmask
        out = decoding.full_attn(queries, keys, values, bmask, self.chunk_size, attn_mask, prior)
        if self.return_active_blocks:
            return out, _count_active_blocks(bmask)
        return out
