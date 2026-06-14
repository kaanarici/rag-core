"""Cached embedding observation and diagnostics models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

EmbeddingOperation = Literal["embed_texts", "embed_query"]
EMBEDDING_OPERATION_TEXTS: Final[EmbeddingOperation] = "embed_texts"
EMBEDDING_OPERATION_QUERY: Final[EmbeddingOperation] = "embed_query"


@dataclass(frozen=True)
class EmbeddingCacheObservation:
    """Cache behavior observed for one embedding call."""

    operation: EmbeddingOperation
    input_count: int
    cache_hits: int
    cache_misses: int
    cache_writes: int
    cache_bypassed: bool = False


@dataclass(frozen=True)
class CachedEmbeddingDiagnostics:
    """Aggregate cache counters plus the last observed call."""

    text_requests: int
    query_requests: int
    cache_hits: int
    cache_misses: int
    cache_writes: int
    query_bypasses: int
    last_observation: EmbeddingCacheObservation | None


__all__ = [
    "CachedEmbeddingDiagnostics",
    "EMBEDDING_OPERATION_QUERY",
    "EMBEDDING_OPERATION_TEXTS",
    "EmbeddingCacheObservation",
]
