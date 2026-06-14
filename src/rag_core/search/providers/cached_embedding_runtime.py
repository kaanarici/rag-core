from __future__ import annotations

from typing import Protocol, cast

from rag_core.search.provider_protocols import EmbeddingProvider, provider_name

from .cached_embedding_observations import (
    EMBEDDING_OPERATION_QUERY,
    EMBEDDING_OPERATION_TEXTS,
    EmbeddingCacheObservation,
)
from .embedding_cache_models import EmbeddingCache, EmbedCacheKey
from .embedding_results import safe_ordered_embedding_vectors


class _BatchEmbeddingCache(Protocol):
    async def get_many(
        self,
        keys: list[EmbedCacheKey],
    ) -> dict[EmbedCacheKey, list[float]]: ...

    async def put_many(
        self,
        items: dict[EmbedCacheKey, list[float]],
    ) -> None: ...


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
        cached = cached_vectors.get(key)
        cached_vector = _valid_cached_vector(cached, inner=inner)
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


async def _get_many(
    cache: EmbeddingCache,
    keys: list[EmbedCacheKey],
) -> dict[EmbedCacheKey, list[float]]:
    get_many = getattr(cache, "get_many", None)
    if callable(get_many):
        batch_cache = cast(_BatchEmbeddingCache, cache)
        return await batch_cache.get_many(keys)
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
        batch_cache = cast(_BatchEmbeddingCache, cache)
        await batch_cache.put_many(items)
        return
    for key, vector in items.items():
        await cache.put(key, vector)


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

    cached = await cache.get(key)
    cached_vector = _valid_cached_vector(cached, inner=inner)
    if cached_vector is not None:
        return cached_vector, EmbeddingCacheObservation(
            operation=EMBEDDING_OPERATION_QUERY,
            input_count=1,
            cache_hits=1,
            cache_misses=0,
            cache_writes=0,
        )

    vector = await inner.embed_query(query)
    vector = _validated_single_vector(vector, inner=inner)
    await cache.put(key, vector)
    return vector, EmbeddingCacheObservation(
        operation=EMBEDDING_OPERATION_QUERY,
        input_count=1,
        cache_hits=0,
        cache_misses=1,
        cache_writes=1,
    )


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
