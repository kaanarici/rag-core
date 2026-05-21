"""Embedding-provider wrapper that consults an :class:`EmbeddingCache`.

The wrapper preserves the :class:`rag_core.search.types.EmbeddingProvider`
shape so callers swap it in transparently. Query embeddings skip the cache by
default because queries are typically unique per request; corpus-side
``embed_texts`` calls are where reuse pays off.
"""

from __future__ import annotations

from rag_core.search.types import EmbeddingProvider

from .cached_embedding_observations import (
    CachedEmbeddingDiagnostics,
    EmbeddingCacheObservation,
)
from .cached_embedding_state import (
    CachedEmbeddingCounters,
    CachedEmbeddingKeyBuilder,
    EmbeddingInputType,
    Hasher,
    cached_embedding_provider_fingerprint,
    cached_embedding_provider_name,
)
from .cached_embedding_runtime import embed_query_with_cache, embed_texts_with_cache
from .embedding_cache_models import EmbeddingCache, EmbedCacheKey, sha256_text

DEFAULT_EMBEDDING_NORMALIZATION = "text_sha256_utf8"


class CachedEmbeddingProvider:
    """Wrap an embedding provider with an :class:`EmbeddingCache`.

    Cache hits short-circuit the inner provider entirely; misses are batched
    into a single inner ``embed_texts`` call so the underlying API is still
    used efficiently. Cache identity includes the provider, provider
    configuration fingerprint, model, dimensions, input type, normalization,
    processing fingerprint, and text hash.
    """

    def __init__(
        self,
        inner: EmbeddingProvider,
        cache: EmbeddingCache,
        *,
        hasher: Hasher = sha256_text,
        normalization: str = DEFAULT_EMBEDDING_NORMALIZATION,
        provider_config_fingerprint: str | None = None,
        processing_fingerprint: str = "",
        cache_queries: bool = False,
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._provider_name = cached_embedding_provider_name(inner)
        self._key_builder = CachedEmbeddingKeyBuilder(
            provider=self._provider_name,
            provider_config_fingerprint=cached_embedding_provider_fingerprint(
                inner,
                provider_config_fingerprint,
            ),
            model=inner.model_name,
            dimensions=inner.dimensions,
            normalization=normalization,
            hasher=hasher,
        )
        self._processing_fingerprint = processing_fingerprint
        self._cache_queries = cache_queries
        self._counters = CachedEmbeddingCounters()

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def diagnostics(self) -> CachedEmbeddingDiagnostics:
        """Return an immutable cache-observability snapshot."""
        return self._counters.snapshot()

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors, _ = await self.embed_texts_with_observation(texts)
        return vectors

    async def embed_texts_with_observation(
        self, texts: list[str]
    ) -> tuple[list[list[float]], EmbeddingCacheObservation | None]:
        return await self.embed_texts_with_processing_fingerprint(
            texts,
            processing_fingerprint=self._processing_fingerprint,
        )

    async def embed_texts_with_processing_fingerprint(
        self,
        texts: list[str],
        *,
        processing_fingerprint: str,
    ) -> tuple[list[list[float]], EmbeddingCacheObservation | None]:
        if not texts:
            return [], None
        vectors, observation = await embed_texts_with_cache(
            inner=self._inner,
            cache=self._cache,
            texts=texts,
            keys=[
                self._build_key(
                    text,
                    input_type="document",
                    processing_fingerprint=processing_fingerprint,
                )
                for text in texts
            ],
        )
        self._record_observation(observation)
        return vectors, observation

    async def embed_query(self, query: str) -> list[float]:
        vector, _ = await self.embed_query_with_observation(query)
        return vector

    async def embed_query_with_observation(
        self, query: str
    ) -> tuple[list[float], EmbeddingCacheObservation]:
        key = self._build_key(
            query,
            input_type="query",
            processing_fingerprint=self._processing_fingerprint,
        )
        vector, observation = await embed_query_with_cache(
            inner=self._inner,
            cache=self._cache,
            query=query,
            key=key,
            cache_queries=self._cache_queries,
        )
        self._record_observation(observation)
        return vector, observation

    def _build_key(
        self,
        text: str,
        *,
        input_type: EmbeddingInputType,
        processing_fingerprint: str,
    ) -> EmbedCacheKey:
        return self._key_builder.build(
            text,
            input_type=input_type,
            processing_fingerprint=processing_fingerprint,
        )

    def _record_observation(self, observation: EmbeddingCacheObservation) -> None:
        self._counters.record(observation)


__all__ = [
    "CachedEmbeddingDiagnostics",
    "CachedEmbeddingProvider",
    "EmbeddingCacheObservation",
]
