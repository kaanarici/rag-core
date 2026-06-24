"""Embedding-provider wrapper that consults an :class:`EmbeddingCache`.

The wrapper preserves the :class:`rag_core.search.provider_protocols.EmbeddingProvider`
shape so callers swap it in transparently. Query embeddings skip the cache by
default because queries are typically unique per request; collection-side
``embed_texts`` calls are where reuse pays off.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal, Protocol, cast

from rag_core.search.provider_protocols import EmbeddingProvider, provider_name

from .embedding_cache_models import (
    EMPTY_EMBEDDING_DOCUMENT_SCOPE,
    EmbeddingCache,
    EmbeddingDocumentScope,
    EmbedCacheKey,
    sha256_text,
)
from .embedding_results import safe_ordered_embedding_vectors
from .embedding_input_types import (
    EMBEDDING_INPUT_DOCUMENT,
    EMBEDDING_INPUT_QUERY,
    EmbeddingInputType,
)

DEFAULT_EMBEDDING_NORMALIZATION = "text_sha256_utf8"
EmbeddingOperation = Literal["embed_texts", "embed_query"]
EMBEDDING_OPERATION_TEXTS: Final[EmbeddingOperation] = "embed_texts"
EMBEDDING_OPERATION_QUERY: Final[EmbeddingOperation] = "embed_query"
Hasher = Callable[[str], str]


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


@dataclass(frozen=True)
class _KeyBuilder:
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
        scope: EmbeddingDocumentScope = EMPTY_EMBEDDING_DOCUMENT_SCOPE,
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
            namespace=scope.namespace,
            collection=scope.collection,
            document_id=scope.document_id,
        )


@dataclass
class _Counters:
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


class _BatchEmbeddingCache(Protocol):
    async def get_many(
        self,
        keys: list[EmbedCacheKey],
    ) -> dict[EmbedCacheKey, list[float]]: ...

    async def put_many(
        self,
        items: dict[EmbedCacheKey, list[float]],
    ) -> None: ...


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
        self._provider_name = provider_name(inner)
        self._key_builder = _KeyBuilder(
            provider=self._provider_name,
            provider_config_fingerprint=_provider_fingerprint(
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
        self._counters = _Counters()

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
        scope: EmbeddingDocumentScope = EMPTY_EMBEDDING_DOCUMENT_SCOPE,
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
                    input_type=EMBEDDING_INPUT_DOCUMENT,
                    processing_fingerprint=processing_fingerprint,
                    scope=scope,
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
            input_type=EMBEDDING_INPUT_QUERY,
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
        scope: EmbeddingDocumentScope = EMPTY_EMBEDDING_DOCUMENT_SCOPE,
    ) -> EmbedCacheKey:
        return self._key_builder.build(
            text,
            input_type=input_type,
            processing_fingerprint=processing_fingerprint,
            scope=scope,
        )

    def _record_observation(self, observation: EmbeddingCacheObservation) -> None:
        self._counters.record(observation)


async def embed_texts_with_cache(
    *,
    inner: EmbeddingProvider,
    cache: EmbeddingCache,
    texts: list[str],
    keys: list[EmbedCacheKey],
) -> tuple[list[list[float]], EmbeddingCacheObservation]:
    results: list[list[float] | None] = []
    miss_key_order: list[EmbedCacheKey] = []
    miss_indices_by_key: dict[EmbedCacheKey, list[int]] = {}
    miss_text_by_key: dict[EmbedCacheKey, str] = {}
    cache_hits = 0
    cache_misses = 0
    cached_vectors = await _get_many(cache, keys)
    for index, (text, key) in enumerate(zip(texts, keys, strict=True)):
        cached_vector = _valid_cached_vector(cached_vectors.get(key), inner=inner)
        if cached_vector is not None:
            results.append(cached_vector)
            cache_hits += 1
            continue
        results.append(None)
        if key not in miss_indices_by_key:
            miss_indices_by_key[key] = []
            miss_text_by_key[key] = text
            miss_key_order.append(key)
        miss_indices_by_key[key].append(index)
        cache_misses += 1

    cache_writes = 0
    if miss_key_order:
        miss_texts = [miss_text_by_key[key] for key in miss_key_order]
        miss_vectors = await _embed_text_misses(inner, miss_texts)
        cache_items: dict[EmbedCacheKey, list[float]] = {}
        for key, vector in zip(miss_key_order, miss_vectors, strict=True):
            vector_copy = list(vector)
            for offset in miss_indices_by_key[key]:
                results[offset] = vector_copy
            cache_items[key] = vector_copy
            cache_writes += 1
        await _put_many(cache, cache_items)

    return _filled_vectors(results), EmbeddingCacheObservation(
        operation=EMBEDDING_OPERATION_TEXTS,
        input_count=len(texts),
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        cache_writes=cache_writes,
    )


async def embed_query_with_cache(
    *,
    inner: EmbeddingProvider,
    cache: EmbeddingCache,
    query: str,
    key: EmbedCacheKey,
    cache_queries: bool,
) -> tuple[list[float], EmbeddingCacheObservation]:
    if not cache_queries:
        vector = await _embed_single_query(inner, query)
        return vector, EmbeddingCacheObservation(
            operation=EMBEDDING_OPERATION_QUERY,
            input_count=1,
            cache_hits=0,
            cache_misses=0,
            cache_writes=0,
            cache_bypassed=True,
        )

    cached_vector = _valid_cached_vector(await cache.get(key), inner=inner)
    if cached_vector is not None:
        return cached_vector, EmbeddingCacheObservation(
            operation=EMBEDDING_OPERATION_QUERY,
            input_count=1,
            cache_hits=1,
            cache_misses=0,
            cache_writes=0,
        )

    vector = _validated_single_vector(await inner.embed_query(query), inner=inner)
    await cache.put(key, vector)
    return vector, EmbeddingCacheObservation(
        operation=EMBEDDING_OPERATION_QUERY,
        input_count=1,
        cache_hits=0,
        cache_misses=1,
        cache_writes=1,
    )


async def _get_many(
    cache: EmbeddingCache,
    keys: list[EmbedCacheKey],
) -> dict[EmbedCacheKey, list[float]]:
    get_many = getattr(cache, "get_many", None)
    if callable(get_many):
        return await cast(_BatchEmbeddingCache, cache).get_many(keys)
    cached_vectors: dict[EmbedCacheKey, list[float]] = {}
    for key in keys:
        cached = await cache.get(key)
        if cached is not None:
            cached_vectors[key] = cached
    return cached_vectors


async def _put_many(
    cache: EmbeddingCache,
    items: dict[EmbedCacheKey, list[float]],
) -> None:
    if not items:
        return
    put_many = getattr(cache, "put_many", None)
    if callable(put_many):
        await cast(_BatchEmbeddingCache, cache).put_many(items)
        return
    for key, vector in items.items():
        await cache.put(key, vector)


def _valid_cached_vector(
    cached: list[float] | None,
    *,
    inner: EmbeddingProvider,
) -> list[float] | None:
    if cached is None:
        return None
    try:
        return safe_ordered_embedding_vectors(
            rows=[cached],
            expected_count=1,
            expected_dimensions=inner.dimensions,
            provider_name=f"{provider_name(inner)} cache",
        )[0]
    except ValueError:
        return None


async def _embed_text_misses(
    inner: EmbeddingProvider,
    texts: list[str],
) -> list[list[float]]:
    vectors = await inner.embed_texts(texts)
    return safe_ordered_embedding_vectors(
        rows=list(vectors),
        expected_count=len(texts),
        expected_dimensions=inner.dimensions,
        provider_name=provider_name(inner),
    )


async def _embed_single_query(
    inner: EmbeddingProvider,
    query: str,
) -> list[float]:
    return _validated_single_vector(await inner.embed_query(query), inner=inner)


def _validated_single_vector(
    vector: list[float],
    *,
    inner: EmbeddingProvider,
) -> list[float]:
    return safe_ordered_embedding_vectors(
        rows=[vector],
        expected_count=1,
        expected_dimensions=inner.dimensions,
        provider_name=provider_name(inner),
    )[0]


def _filled_vectors(results: list[list[float] | None]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for result_index, vector in enumerate(results):
        if vector is None:
            raise ValueError(
                "embedding cache runtime did not fill result index %d" % result_index
            )
        vectors.append(vector)
    return vectors


def _provider_fingerprint(
    inner: EmbeddingProvider,
    explicit: str | None,
) -> str:
    if explicit is not None:
        return explicit
    return str(getattr(inner, "cache_identity", ""))


__all__ = [
    "CachedEmbeddingDiagnostics",
    "CachedEmbeddingProvider",
    "EmbeddingCacheObservation",
]
