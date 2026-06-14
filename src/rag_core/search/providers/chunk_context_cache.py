from __future__ import annotations

import asyncio
import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol, TypeVar, runtime_checkable

from rag_core.search.providers.cache_provider_names import (
    IN_MEMORY_CACHE_PROVIDER,
    NO_CACHE_PROVIDER,
    SQLITE_CACHE_PROVIDER,
)
from rag_core.search.providers.cache_sqlite import (
    evict_sqlite_cache_rows,
    sqlite_table_schema,
    validate_sqlite_cache_bounds,
)
from rag_core.search.providers.cache_sqlite_connection import (
    SqliteCacheConnectionPool,
    bind_running_task_for_pool,
    harden_sqlite_sidecar_files,
)
from rag_core.search.providers.chunk_context_sqlite_cache import (
    ensure_chunk_context_cache_eviction_index,
    execute_delete_by_scope as _execute_delete_by_scope,
    execute_put as _execute_put,
    executemany_put as _executemany_put,
    fetch_chunk_context_batch as _fetch_chunk_context_batch,
    fetchone as _fetchone,
)
from rag_core.search.providers.registry import CHUNK_CONTEXT_CACHES

_RunResult = TypeVar("_RunResult")


@dataclass(frozen=True)
class ChunkContextKey:
    """Key for a chunk-context cache entry.

    Scope (``namespace`` / ``corpus_id`` / ``document_id``) is part of the
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
    corpus_id: str
    document_id: str

    def stringify(self) -> str:
        return (
            f"{self.contextualizer_id}|{self.document_sha256}|"
            f"{self.document_filename_sha256}|{self.chunk_text_sha256}|"
            f"{self.chunk_index}|{self.total_chunks}|"
            f"{self.namespace}|{self.corpus_id}|{self.document_id}"
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
        corpus_id: str,
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
        corpus_id: str,
        document_id: str,
    ) -> int:
        del namespace, corpus_id, document_id
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
            (key.namespace, key.corpus_id, key.document_id), set()
        ).add(cache_key)

    async def put_many(self, items: dict[ChunkContextKey, str]) -> None:
        for key, context in items.items():
            cache_key = key.stringify()
            self._store[cache_key] = context
            self._scope_index.setdefault(
                (key.namespace, key.corpus_id, key.document_id), set()
            ).add(cache_key)

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


class SqliteChunkContextCache:
    """Disk-backed chunk-context cache.

    Optional entry bounds use FIFO-by-write timestamp; cache hits do not
    refresh ``ts``.
    """

    provider_name = SQLITE_CACHE_PROVIDER

    _CREATE_SQL = (
        "CREATE TABLE IF NOT EXISTS chunk_context_cache ("
        "key TEXT PRIMARY KEY, context TEXT NOT NULL, "
        "contextualizer_id TEXT NOT NULL, "
        "namespace TEXT NOT NULL DEFAULT '', "
        "corpus_id TEXT NOT NULL DEFAULT '', "
        "document_id TEXT NOT NULL DEFAULT '', "
        "ts INTEGER NOT NULL"
        ")"
    )
    _CREATE_SCOPE_INDEX_SQL = (
        "CREATE INDEX IF NOT EXISTS idx_chunk_context_cache_scope "
        "ON chunk_context_cache (namespace, corpus_id, document_id)"
    )
    _EXPECTED_SCHEMA: ClassVar[frozenset[tuple[str, str, int, int]]] = frozenset(
        {
            ("key", "TEXT", 0, 1),
            ("context", "TEXT", 1, 0),
            ("contextualizer_id", "TEXT", 1, 0),
            ("namespace", "TEXT", 1, 0),
            ("corpus_id", "TEXT", 1, 0),
            ("document_id", "TEXT", 1, 0),
            ("ts", "INTEGER", 1, 0),
        }
    )

    def __init__(
        self,
        path: str | Path,
        *,
        max_age_seconds: float | None = None,
        max_entries: int | None = None,
    ) -> None:
        validate_sqlite_cache_bounds(
            max_age_seconds=max_age_seconds,
            max_entries=max_entries,
        )
        self._path = str(path)
        self._pool = SqliteCacheConnectionPool(self._path)
        self._schema_ready: set[int] = set()
        self._max_age_seconds = max_age_seconds
        self._max_entries = max_entries
        self._eviction_enabled = max_age_seconds is not None or max_entries is not None

    @property
    def path(self) -> str:
        return self._path

    def _prepare_connection(self, connection: sqlite3.Connection) -> None:
        connection_id = id(connection)
        if connection_id not in self._schema_ready:
            self._ensure_schema(connection)
            if self._eviction_enabled:
                ensure_chunk_context_cache_eviction_index(connection)
                evict_sqlite_cache_rows(
                    connection,
                    table_name="chunk_context_cache",
                    max_age_seconds=self._max_age_seconds,
                    max_entries=self._max_entries,
                )
            connection.commit()
            harden_sqlite_sidecar_files(Path(self._path))
            self._schema_ready.add(connection_id)

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
        schema = sqlite_table_schema(connection, "chunk_context_cache")
        if schema and schema != self._EXPECTED_SCHEMA:
            connection.execute("DROP TABLE chunk_context_cache")
        connection.execute(self._CREATE_SQL)
        connection.execute(self._CREATE_SCOPE_INDEX_SQL)

    async def _run(
        self,
        func: Callable[..., _RunResult],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> _RunResult:
        # ``asyncio.to_thread`` copies the caller's contextvars into the
        # worker; binding the running task here lets the pool inside the
        # worker recover the originating task and hand back the same
        # per-task connection across many SQL calls.
        bind_running_task_for_pool(self._pool)

        def _bound() -> _RunResult:
            with self._pool.connection() as connection:
                self._prepare_connection(connection)
                return func(connection, *args, **kwargs)

        return await asyncio.to_thread(_bound)

    async def get(self, key: ChunkContextKey) -> str | None:
        row = await self._run(
            _fetchone,
            "SELECT context FROM chunk_context_cache WHERE key = ?",
            (key.stringify(),),
        )
        return None if row is None else str(row[0])

    async def get_many(self, keys: list[ChunkContextKey]) -> dict[ChunkContextKey, str]:
        stringified_keys = [key.stringify() for key in keys]
        contexts = await self._run(
            _fetch_chunk_context_batch,
            list(dict.fromkeys(stringified_keys)),
        )
        return {
            key: contexts[cache_key]
            for key in keys
            if (cache_key := key.stringify()) in contexts
        }

    async def put(self, key: ChunkContextKey, context: str) -> None:
        timestamp = int(time.time())
        await self._run(
            _execute_put,
            "INSERT OR REPLACE INTO chunk_context_cache "
            "(key, context, contextualizer_id, namespace, corpus_id, document_id, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                key.stringify(),
                context,
                key.contextualizer_id,
                key.namespace,
                key.corpus_id,
                key.document_id,
                timestamp,
            ),
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
                key.corpus_id,
                key.document_id,
                timestamp,
            )
            for key, context in items.items()
        ]
        await self._run(
            _executemany_put,
            "INSERT OR REPLACE INTO chunk_context_cache "
            "(key, context, contextualizer_id, namespace, corpus_id, document_id, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
            max_age_seconds=self._max_age_seconds,
            max_entries=self._max_entries,
            now=timestamp,
        )

    async def delete_by_document_scope(
        self,
        *,
        namespace: str,
        corpus_id: str,
        document_id: str,
    ) -> int:
        deleted: int = await self._run(
            _execute_delete_by_scope,
            namespace,
            corpus_id,
            document_id,
        )
        return deleted

    def close(self) -> None:
        self._pool.close()
        self._schema_ready.clear()


def _chunks(values: Sequence[str], size: int) -> list[Sequence[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


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
