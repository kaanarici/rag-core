from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, cast

from rag_core.search.providers.cached_embedding import (
    EmbeddingCacheObservation,
)
from rag_core.search.providers.embedding_cache_models import (
    EMPTY_EMBEDDING_DOCUMENT_SCOPE,
    EmbeddingDocumentScope,
)
from rag_core.search.provider_protocols import EmbeddingProvider

_ObservedEmbedTexts = Callable[
    [list[str]],
    Awaitable[tuple[list[list[float]], EmbeddingCacheObservation | None]],
]
_ObservedEmbedQuery = Callable[
    [str],
    Awaitable[tuple[list[float], EmbeddingCacheObservation]],
]


class _ObservedEmbedTextsWithFingerprint(Protocol):
    async def __call__(
        self,
        texts: list[str],
        *,
        processing_fingerprint: str,
        scope: EmbeddingDocumentScope = ...,
    ) -> tuple[list[list[float]], EmbeddingCacheObservation | None]: ...


@dataclass(frozen=True)
class EmbeddingCacheCounters:
    hits: int = 0
    misses: int = 0
    writes: int = 0
    bypasses: int = 0


async def embed_texts_with_cache_observation(
    provider: EmbeddingProvider,
    texts: list[str],
    *,
    processing_fingerprint: str = "",
    scope: EmbeddingDocumentScope = EMPTY_EMBEDDING_DOCUMENT_SCOPE,
) -> tuple[list[list[float]], EmbeddingCacheCounters]:
    observed_embed_with_fingerprint = getattr(
        provider,
        "embed_texts_with_processing_fingerprint",
        None,
    )
    if callable(observed_embed_with_fingerprint):
        vectors, observation = await cast(
            _ObservedEmbedTextsWithFingerprint,
            observed_embed_with_fingerprint,
        )(texts, processing_fingerprint=processing_fingerprint, scope=scope)
        return vectors, _counters_from_observation(observation)
    observed_embed = getattr(provider, "embed_texts_with_observation", None)
    if callable(observed_embed):
        vectors, observation = await cast(_ObservedEmbedTexts, observed_embed)(texts)
        return vectors, _counters_from_observation(observation)
    return await provider.embed_texts(texts), EmbeddingCacheCounters()


async def embed_query_with_cache_observation(
    provider: EmbeddingProvider, query: str
) -> tuple[list[float], EmbeddingCacheCounters]:
    observed_embed = getattr(provider, "embed_query_with_observation", None)
    if callable(observed_embed):
        vector, observation = await cast(_ObservedEmbedQuery, observed_embed)(query)
        return vector, _counters_from_observation(observation)
    return await provider.embed_query(query), EmbeddingCacheCounters()


def _counters_from_observation(
    observation: EmbeddingCacheObservation | None,
) -> EmbeddingCacheCounters:
    if observation is None:
        return EmbeddingCacheCounters()
    return EmbeddingCacheCounters(
        hits=observation.cache_hits,
        misses=observation.cache_misses,
        writes=observation.cache_writes,
        bypasses=1 if observation.cache_bypassed else 0,
    )
