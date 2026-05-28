"""No-op and in-memory embedding cache implementations."""

from __future__ import annotations

from collections import OrderedDict

from rag_core.search.providers.cache_provider_names import (
    IN_MEMORY_CACHE_PROVIDER,
    NO_CACHE_PROVIDER,
)
from rag_core.search.providers.embedding_cache_models import EmbedCacheKey


class NoCache:
    """Default cache that always misses."""

    provider_name = NO_CACHE_PROVIDER

    async def get(self, key: EmbedCacheKey) -> list[float] | None:
        return None

    async def put(self, key: EmbedCacheKey, vector: list[float]) -> None:
        return None


class InMemoryCache:
    """LRU dict-backed embedding cache."""

    provider_name = IN_MEMORY_CACHE_PROVIDER

    def __init__(self, *, max_entries: int | None = None) -> None:
        if max_entries is not None and max_entries <= 0:
            raise ValueError("max_entries must be positive or None")
        self._max_entries = max_entries
        self._store: OrderedDict[str, list[float]] = OrderedDict()

    async def get(self, key: EmbedCacheKey) -> list[float] | None:
        cache_key = key.stringify()
        if cache_key not in self._store:
            return None
        value = self._store.pop(cache_key)
        self._store[cache_key] = value
        return list(value)

    async def put(self, key: EmbedCacheKey, vector: list[float]) -> None:
        cache_key = key.stringify()
        if cache_key in self._store:
            self._store.pop(cache_key)
        self._store[cache_key] = list(vector)
        if (
            self._max_entries is not None
            and len(self._store) > self._max_entries
        ):
            self._store.popitem(last=False)


__all__ = [
    "InMemoryCache",
    "NoCache",
]
