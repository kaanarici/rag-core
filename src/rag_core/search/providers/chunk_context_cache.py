from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

from rag_core.search.providers.cache_sqlite import (
    IN_MEMORY_CACHE_PROVIDER,
    NO_CACHE_PROVIDER,
    SQLITE_CACHE_PROVIDER,
    SqliteCacheBase,
    SqliteCacheSchema,
    delete_cache_rows_by_scope,
    executemany_with_eviction,
    execute_write_with_eviction,
    fetch_keyed_text_values,
    fetch_one,
)
from rag_core.search.providers.registry import CHUNK_CONTEXT_CACHES

CREATE_CHUNK_CONTEXT_CACHE_SQL = (
    "CREATE TABLE IF NOT EXISTS chunk_context_cache ("
    "key TEXT PRIMARY KEY, context TEXT NOT NULL, "
    "contextualizer_id TEXT NOT NULL, "
    "namespace TEXT NOT NULL DEFAULT '', "
    "collection TEXT NOT NULL DEFAULT '', "
    "document_id TEXT NOT NULL DEFAULT '', "
    "ts INTEGER NOT NULL"
    ")"
)
CREATE_CHUNK_CONTEXT_CACHE_SCOPE_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_chunk_context_cache_scope "
    "ON chunk_context_cache (namespace, collection, document_id)"
)
CREATE_CHUNK_CONTEXT_CACHE_TS_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_chunk_context_cache_ts "
    "ON chunk_context_cache (ts, key)"
)
EXPECTED_CHUNK_CONTEXT_CACHE_SCHEMA: frozenset[tuple[str, str, int, int]] = frozenset(
    {
        ("key", "TEXT", 0, 1),
        ("context", "TEXT", 1, 0),
        ("contextualizer_id", "TEXT", 1, 0),
        ("namespace", "TEXT", 1, 0),
        ("collection", "TEXT", 1, 0),
        ("document_id", "TEXT", 1, 0),
        ("ts", "INTEGER", 1, 0),
    }
)
_CHUNK_CONTEXT_SCHEMA = SqliteCacheSchema(
    table_name="chunk_context_cache",
    create_table_sql=CREATE_CHUNK_CONTEXT_CACHE_SQL,
    expected_schema=EXPECTED_CHUNK_CONTEXT_CACHE_SCHEMA,
    create_scope_index_sql=CREATE_CHUNK_CONTEXT_CACHE_SCOPE_INDEX_SQL,
    create_eviction_index_sql=CREATE_CHUNK_CONTEXT_CACHE_TS_INDEX_SQL,
)


@dataclass(frozen=True)
class ChunkContextKey:
    """Key for a chunk-context cache entry.

    Scope (``namespace`` / ``collection`` / ``document_id``) is part of the
    key so a ``delete_by_document_scope`` purge clears every cached
    contextualization derived from the deleted document. Direct prepare calls
    that are not tied to an ingest identity pass explicit empty scope strings.
    """

    contextualizer_id: str
    document_sha256: str
    document_filename_sha256: str
    chunk_text_sha256: str
    chunk_index: int
    total_chunks: int
    namespace: str
    collection: str
    document_id: str

    def stringify(self) -> str:
        return (
            f"{self.contextualizer_id}|{self.document_sha256}|"
            f"{self.document_filename_sha256}|{self.chunk_text_sha256}|"
            f"{self.chunk_index}|{self.total_chunks}|"
            f"{self.namespace}|{self.collection}|{self.document_id}"
        )


@runtime_checkable
class ChunkContextCache(Protocol):
    """Look up and store contextualizer outputs."""

    async def get(self, key: ChunkContextKey) -> str | None: ...

    async def get_many(self, keys: list[ChunkContextKey]) -> dict[ChunkContextKey, str]: ...

    async def put(self, key: ChunkContextKey, context: str) -> None: ...

    async def put_many(self, items: dict[ChunkContextKey, str]) -> None: ...


@runtime_checkable
class ScopedDeletableChunkContextCache(Protocol):
    """Optional capability for caches that support scope-based purge.

    Probed via ``getattr`` by the delete facade so a third-party cache that
    cannot scope-delete is fine. Its
    ``DeleteDocumentResult.chunk_context_cache_purged`` field just stays
    ``None`` and the delete-recovery journal does not retain a partial-state
    entry for that surface.
    """

    async def delete_by_document_scope(
        self,
        *,
        namespace: str,
        collection: str,
        document_id: str,
    ) -> int: ...


class NoChunkContextCache:
    """Default contextualizer cache that always misses."""

    provider_name = NO_CACHE_PROVIDER

    async def get(self, key: ChunkContextKey) -> str | None:
        return None

    async def get_many(self, keys: list[ChunkContextKey]) -> dict[ChunkContextKey, str]:
        return {}

    async def put(self, key: ChunkContextKey, context: str) -> None:
        return None

    async def put_many(self, items: dict[ChunkContextKey, str]) -> None:
        return None

    async def delete_by_document_scope(
        self,
        *,
        namespace: str,
        collection: str,
        document_id: str,
    ) -> int:
        del namespace, collection, document_id
        return 0


class InMemoryChunkContextCache:
    """Dict-backed chunk-context cache with scope-keyed reverse index."""

    provider_name = IN_MEMORY_CACHE_PROVIDER

    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self._scope_index: dict[tuple[str, str, str], set[str]] = {}

    async def get(self, key: ChunkContextKey) -> str | None:
        return self._store.get(key.stringify())

    async def get_many(self, keys: list[ChunkContextKey]) -> dict[ChunkContextKey, str]:
        return {
            key: self._store[cache_key]
            for key in keys
            if (cache_key := key.stringify()) in self._store
        }

    async def put(self, key: ChunkContextKey, context: str) -> None:
        cache_key = key.stringify()
        self._store[cache_key] = context
        self._scope_index.setdefault(
            (key.namespace, key.collection, key.document_id), set()
        ).add(cache_key)

    async def put_many(self, items: dict[ChunkContextKey, str]) -> None:
        for key, context in items.items():
            cache_key = key.stringify()
            self._store[cache_key] = context
            self._scope_index.setdefault(
                (key.namespace, key.collection, key.document_id), set()
            ).add(cache_key)

    async def delete_by_document_scope(
        self,
        *,
        namespace: str,
        collection: str,
        document_id: str,
    ) -> int:
        scope = (namespace, collection, document_id)
        cache_keys = self._scope_index.pop(scope, None)
        if not cache_keys:
            return 0
        removed = 0
        for cache_key in cache_keys:
            if self._store.pop(cache_key, None) is not None:
                removed += 1
        return removed


class SqliteChunkContextCache(SqliteCacheBase):
    """Disk-backed chunk-context cache.

    Optional entry bounds use FIFO-by-write timestamp; cache hits do not
    refresh ``ts``.
    """

    provider_name = SQLITE_CACHE_PROVIDER
    _SCHEMA: ClassVar[SqliteCacheSchema] = _CHUNK_CONTEXT_SCHEMA
    _EXPECTED_SCHEMA: ClassVar[frozenset[tuple[str, str, int, int]]] = frozenset(
        EXPECTED_CHUNK_CONTEXT_CACHE_SCHEMA
    )

    async def get(self, key: ChunkContextKey) -> str | None:
        row = await self._run(
            fetch_one,
            "SELECT context FROM chunk_context_cache WHERE key = ?",
            (key.stringify(),),
        )
        return None if row is None else str(row[0])

    async def get_many(self, keys: list[ChunkContextKey]) -> dict[ChunkContextKey, str]:
        stringified_keys = [key.stringify() for key in keys]
        contexts = await self._run(
            fetch_keyed_text_values,
            table_name="chunk_context_cache",
            value_column="context",
            cache_keys=list(dict.fromkeys(stringified_keys)),
        )
        return {
            key: contexts[cache_key]
            for key in keys
            if (cache_key := key.stringify()) in contexts
        }

    async def put(self, key: ChunkContextKey, context: str) -> None:
        timestamp = int(time.time())
        await self._run(
            execute_write_with_eviction,
            sql=(
                "INSERT OR REPLACE INTO chunk_context_cache "
                "(key, context, contextualizer_id, namespace, collection, document_id, ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)"
            ),
            params=(
                key.stringify(),
                context,
                key.contextualizer_id,
                key.namespace,
                key.collection,
                key.document_id,
                timestamp,
            ),
            table_name="chunk_context_cache",
            max_age_seconds=self._max_age_seconds,
            max_entries=self._max_entries,
            now=timestamp,
        )

    async def put_many(self, items: dict[ChunkContextKey, str]) -> None:
        if not items:
            return
        timestamp = int(time.time())
        rows = [
            (
                key.stringify(),
                context,
                key.contextualizer_id,
                key.namespace,
                key.collection,
                key.document_id,
                timestamp,
            )
            for key, context in items.items()
        ]
        await self._run(
            executemany_with_eviction,
            sql=(
                "INSERT OR REPLACE INTO chunk_context_cache "
                "(key, context, contextualizer_id, namespace, collection, document_id, ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)"
            ),
            rows=rows,
            table_name="chunk_context_cache",
            max_age_seconds=self._max_age_seconds,
            max_entries=self._max_entries,
            now=timestamp,
        )

    async def delete_by_document_scope(
        self,
        *,
        namespace: str,
        collection: str,
        document_id: str,
    ) -> int:
        deleted: int = await self._run(
            delete_cache_rows_by_scope,
            table_name="chunk_context_cache",
            namespace=namespace,
            collection=collection,
            document_id=document_id,
        )
        return deleted


DEFAULT_CHUNK_CONTEXT_CACHE_PROVIDER = NO_CACHE_PROVIDER


def _build_no_chunk_context_cache(**_: object) -> NoChunkContextCache:
    return NoChunkContextCache()


def _build_in_memory_chunk_context_cache(**_: object) -> InMemoryChunkContextCache:
    return InMemoryChunkContextCache()


def _build_sqlite_chunk_context_cache(**kwargs: Any) -> SqliteChunkContextCache:
    return SqliteChunkContextCache(**kwargs)


def create_chunk_context_cache(
    provider: str | None = None,
    **kwargs: Any,
) -> ChunkContextCache:
    """Resolve the ChunkContextCache provider category from a config name.

    ``None`` resolves to ``"none"`` (the no-op chunk-context cache).
    """
    return CHUNK_CONTEXT_CACHES.create(
        provider or DEFAULT_CHUNK_CONTEXT_CACHE_PROVIDER,
        **kwargs,
    )


CHUNK_CONTEXT_CACHES.register(
    DEFAULT_CHUNK_CONTEXT_CACHE_PROVIDER,
    _build_no_chunk_context_cache,
)
CHUNK_CONTEXT_CACHES.register(
    InMemoryChunkContextCache.provider_name,
    _build_in_memory_chunk_context_cache,
)
CHUNK_CONTEXT_CACHES.register(
    SqliteChunkContextCache.provider_name,
    _build_sqlite_chunk_context_cache,
)


__all__ = [
    "ChunkContextCache",
    "ChunkContextKey",
    "DEFAULT_CHUNK_CONTEXT_CACHE_PROVIDER",
    "InMemoryChunkContextCache",
    "NoChunkContextCache",
    "ScopedDeletableChunkContextCache",
    "SqliteChunkContextCache",
    "create_chunk_context_cache",
]
