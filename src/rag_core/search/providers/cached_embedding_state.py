from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from rag_core.search.provider_protocols import EmbeddingProvider

from .cached_embedding_observations import (
    CachedEmbeddingDiagnostics,
    EMBEDDING_OPERATION_TEXTS,
    EmbeddingCacheObservation,
)
from .embedding_input_types import EmbeddingInputType
from .embedding_cache_models import EmbedCacheKey

Hasher = Callable[[str], str]


@dataclass(frozen=True)
class CachedEmbeddingKeyBuilder:
    provider: str
    provider_config_fingerprint: str
    model: str
    dimensions: int
    normalization: str
    hasher: Hasher

    def build(
        self,
        text: str,
        *,
        input_type: EmbeddingInputType,
        processing_fingerprint: str,
    ) -> EmbedCacheKey:
        return EmbedCacheKey(
            provider=self.provider,
            provider_config_fingerprint=self.provider_config_fingerprint,
            model=self.model,
            dimensions=self.dimensions,
            input_type=input_type,
            normalization=self.normalization,
            processing_fingerprint=processing_fingerprint,
            content_sha256=self.hasher(text),
        )


@dataclass
class CachedEmbeddingCounters:
    text_requests: int = 0
    query_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_writes: int = 0
    query_bypasses: int = 0
    last_observation: EmbeddingCacheObservation | None = None

    def snapshot(self) -> CachedEmbeddingDiagnostics:
        return CachedEmbeddingDiagnostics(
            text_requests=self.text_requests,
            query_requests=self.query_requests,
            cache_hits=self.cache_hits,
            cache_misses=self.cache_misses,
            cache_writes=self.cache_writes,
            query_bypasses=self.query_bypasses,
            last_observation=self.last_observation,
        )

    def record(self, observation: EmbeddingCacheObservation) -> None:
        self.cache_hits += observation.cache_hits
        self.cache_misses += observation.cache_misses
        self.cache_writes += observation.cache_writes
        if observation.operation == EMBEDDING_OPERATION_TEXTS:
            self.text_requests += 1
        else:
            self.query_requests += 1
            if observation.cache_bypassed:
                self.query_bypasses += 1
        self.last_observation = observation


def cached_embedding_provider_fingerprint(
    inner: EmbeddingProvider,
    explicit: str | None,
) -> str:
    if explicit is not None:
        return explicit
    return str(getattr(inner, "cache_identity", ""))


__all__ = [
    "CachedEmbeddingCounters",
    "CachedEmbeddingKeyBuilder",
    "Hasher",
    "cached_embedding_provider_fingerprint",
]
