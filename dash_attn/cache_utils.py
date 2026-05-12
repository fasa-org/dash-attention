import torch


class ChunkRepresentationCache:
    """
    Static pre-allocated cache for chunk representations.

    The buffer `[B, Hkv, max_chunks, D]` is allocated once per sequence by
    `allocate(...)` and reused across decode steps via in-place writes. The
    tensor identity of `_buffer` never changes after allocation

    `n_chunks` is a int tracking the valid extent.
    """

    def __init__(self, max_chunks: int = 512):
        self.max_chunks = max_chunks
        self._buffer: torch.Tensor | None = None
        self.n_chunks: int = 0

    @property
    def is_initialized(self) -> bool:
        return self._buffer is not None

    @property
    def keys_chunks(self) -> torch.Tensor:
        if self._buffer is None:
            return torch.empty(0)
        return self._buffer[..., : self.n_chunks, :]

    def view(self) -> torch.Tensor:
        return self.keys_chunks

    def allocate(self, B: int, Hkv: int, D: int,
                 dtype: torch.dtype, device: torch.device) -> None:
        needs_alloc = (
            self._buffer is None
            or self._buffer.shape != (B, Hkv, self.max_chunks, D)
            or self._buffer.dtype != dtype
            or self._buffer.device != device
        )
        if needs_alloc:
            self._buffer = torch.zeros(
                B, Hkv, self.max_chunks, D, dtype=dtype, device=device
            )
        self.n_chunks = 0

    def append(self, last_chunk: torch.Tensor) -> None:
        """In-place write of one chunk. last_chunk shape: [B, Hkv, 1, D]."""
        assert self._buffer is not None, "allocate() must be called before append()"
        assert self.n_chunks < self.max_chunks, (
            f"chunk cache overflow: n_chunks={self.n_chunks} max={self.max_chunks}"
        )
        self._buffer[..., self.n_chunks, :].copy_(last_chunk.squeeze(-2))
        self.n_chunks += 1

    def append_many(self, chunks: torch.Tensor) -> None:
        """In-place bulk write. chunks shape: [B, Hkv, k, D]. k may be 0."""
        k = chunks.size(-2)
        if k == 0:
            return
        assert self._buffer is not None, "allocate() must be called before append_many()"
        assert self.n_chunks + k <= self.max_chunks, (
            f"chunk cache overflow: n_chunks={self.n_chunks} + {k} > max={self.max_chunks}"
        )
        self._buffer[..., self.n_chunks : self.n_chunks + k, :].copy_(chunks)
        self.n_chunks += k

    def reset(self) -> None:
        self.n_chunks = 0