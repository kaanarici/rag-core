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

    async def delete_by_document_scope(
        self,
        *,
        namespace: str,
        corpus_id: str,
        document_id: str,
    ) -> int:
        # A NoCache never stored anything; right-to-forget completes trivially.
        del namespace, corpus_id, document_id
        return 0


class InMemoryCache:
    """LRU dict-backed embedding cache with scope-keyed reverse index."""

    provider_name = IN_MEMORY_CACHE_PROVIDER

    def __init__(self, *, max_entries: int | None = None) -> None:
        if max_entries is not None and max_entries <= 0:
            raise ValueError("max_entries must be positive or None")
        self._max_entries = max_entries
        self._store: OrderedDict[str, list[float]] = OrderedDict()
        # ``(namespace, corpus_id, document_id) -> {cache_key, ...}`` so
        # ``delete_by_document_scope`` is O(scoped entries) rather than O(all).
        self._scope_index: dict[tuple[str, str, str], set[str]] = {}

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
        scope = (key.namespace, key.corpus_id, key.document_id)
        self._scope_index.setdefault(scope, set()).add(cache_key)
        if (
            self._max_entries is not None
            and len(self._store) > self._max_entries
        ):
            evicted, _ = self._store.popitem(last=False)
            for keys in self._scope_index.values():
                keys.discard(evicted)

    async def delete_by_document_scope(
        self,
        *,
        namespace: str,
        corpus_id: str,
        document_id: str,
    ) -> int:
        scope = (namespace, corpus_id, document_id)
        cache_keys = self._scope_index.pop(scope, None)
        if not cache_keys:
            return 0
        removed = 0
        for cache_key in cache_keys:
            if self._store.pop(cache_key, None) is not None:
                removed += 1
        return removed


__all__ = [
    "InMemoryCache",
    "NoCache",
]
