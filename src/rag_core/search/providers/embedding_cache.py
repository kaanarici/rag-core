"""Concrete embedding cache providers and cache factory registrations."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, TypeVar

from rag_core.search.providers.cache_provider_names import (
    NO_CACHE_PROVIDER,
    SQLITE_CACHE_PROVIDER,
)
from rag_core.search.providers.cache_sqlite import (
    evict_sqlite_cache_rows,
    validate_sqlite_cache_bounds,
)
from rag_core.search.providers.cache_sqlite_connection import (
    SqliteCacheConnectionPool,
    bind_running_task_for_pool,
    harden_sqlite_sidecar_files,
)
from rag_core.search.providers.registry import (
    EMBEDDING_CACHES,
)
from rag_core.search.providers.embedding_sqlite_cache import (
    EmbeddingCacheWrite,
    EXPECTED_EMBEDDING_CACHE_SCHEMA,
    delete_embedding_cache_by_scope,
    ensure_embedding_cache_eviction_index,
    ensure_embedding_cache_schema,
    read_embedding_cache_vector,
    read_embedding_cache_vectors,
    write_embedding_cache_vector,
    write_embedding_cache_vectors,
)
from rag_core.search.providers.embedding_cache_models import (
    EmbedCacheKey as _EmbedCacheKey,
    EmbeddingCache as _EmbeddingCache,
)
from rag_core.search.providers.embedding_memory_cache import InMemoryCache, NoCache

_RunResult = TypeVar("_RunResult")


class SqliteCache:
    """Disk-backed embedding cache via stdlib :mod:`sqlite3`.

    Vectors are stored as packed float32 blobs. Concurrency model: each
    asyncio task gets its own ``sqlite3.Connection`` via the connection
    pool, and every SQL call is dispatched through ``asyncio.to_thread`` so
    the blocking read/write never stalls the event loop. WAL +
    ``busy_timeout=5000`` (set in ``open_sqlite_cache``) make concurrent
    writers wait instead of raising. Schema is ensured once per pool open.
    Optional entry bounds use FIFO-by-write timestamp; cache hits do not
    refresh ``ts``.
    """

    provider_name = SQLITE_CACHE_PROVIDER

    _EXPECTED_SCHEMA: ClassVar[frozenset[tuple[str, str, int, int]]] = frozenset(
        EXPECTED_EMBEDDING_CACHE_SCHEMA
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

    def _prepare_connection(self, connection: sqlite3.Connection) -> None:
        connection_id = id(connection)
        if connection_id not in self._schema_ready:
            # Schema check is idempotent (CREATE IF NOT EXISTS); racing
            # acquirers may double-run it, which is harmless. ``set`` updates
            # in CPython are atomic enough for this no-op-on-repeat guard.
            ensure_embedding_cache_schema(connection)
            if self._eviction_enabled:
                ensure_embedding_cache_eviction_index(connection)
                evict_sqlite_cache_rows(
                    connection,
                    table_name="embedding_cache",
                    max_age_seconds=self._max_age_seconds,
                    max_entries=self._max_entries,
                )
            connection.commit()
            harden_sqlite_sidecar_files(Path(self._path))
            self._schema_ready.add(connection_id)

    async def get(self, key: _EmbedCacheKey) -> list[float] | None:
        result: list[float] | None = await self._run(
            read_embedding_cache_vector,
            cache_key=key.stringify(),
        )
        return result

    async def get_many(
        self,
        keys: list[_EmbedCacheKey],
    ) -> dict[_EmbedCacheKey, list[float]]:
        vectors = await self._run(
            read_embedding_cache_vectors,
            cache_keys=[key.stringify() for key in keys],
        )
        return {
            key: vectors[cache_key]
            for key in keys
            if (cache_key := key.stringify()) in vectors
        }

    async def put(self, key: _EmbedCacheKey, vector: list[float]) -> None:
        await self._run(
            write_embedding_cache_vector,
            cache_key=key.stringify(),
            vector=vector,
            provider=key.provider,
            provider_config_fingerprint=key.provider_config_fingerprint,
            model=key.model,
            dimensions=key.dimensions,
            input_type=key.input_type,
            normalization=key.normalization,
            processing_fingerprint=key.processing_fingerprint,
            namespace=key.namespace,
            corpus_id=key.corpus_id,
            document_id=key.document_id,
            max_age_seconds=self._max_age_seconds,
            max_entries=self._max_entries,
        )

    async def put_many(
        self,
        items: dict[_EmbedCacheKey, list[float]],
    ) -> None:
        entries = [
            EmbeddingCacheWrite(
                cache_key=key.stringify(),
                vector=vector,
                provider=key.provider,
                provider_config_fingerprint=key.provider_config_fingerprint,
                model=key.model,
                dimensions=key.dimensions,
                input_type=key.input_type,
                normalization=key.normalization,
                processing_fingerprint=key.processing_fingerprint,
                namespace=key.namespace,
                corpus_id=key.corpus_id,
                document_id=key.document_id,
            )
            for key, vector in items.items()
        ]
        await self._run(
            write_embedding_cache_vectors,
            entries=entries,
            max_age_seconds=self._max_age_seconds,
            max_entries=self._max_entries,
        )

    async def delete_by_document_scope(
        self,
        *,
        namespace: str,
        corpus_id: str,
        document_id: str,
    ) -> int:
        """Purge every cached vector tagged to ``(namespace, corpus_id, document_id)``.

        Returns the number of rows removed so the right-to-forget facade can
        record an honest ``DeleteDocumentResult.embedding_cache_purged``.
        """
        deleted: int = await self._run(
            delete_embedding_cache_by_scope,
            namespace=namespace,
            corpus_id=corpus_id,
            document_id=document_id,
        )
        return deleted

    def close(self) -> None:
        self._pool.close()
        self._schema_ready.clear()


DEFAULT_EMBEDDING_CACHE_PROVIDER = NO_CACHE_PROVIDER


def _build_no_embedding_cache(**_: Any) -> NoCache:
    return NoCache()


def _build_in_memory_embedding_cache(**kwargs: Any) -> InMemoryCache:
    return InMemoryCache(**kwargs)


def _build_sqlite_embedding_cache(**kwargs: Any) -> SqliteCache:
    return SqliteCache(**kwargs)


def create_embedding_cache(
    provider: str | None = None,
    **kwargs: Any,
) -> _EmbeddingCache:
    """Resolve the EmbeddingCache provider category from a config name.

    ``None`` resolves to ``"none"`` (the no-op :class:`NoCache`).
    """
    return EMBEDDING_CACHES.create(
        provider or DEFAULT_EMBEDDING_CACHE_PROVIDER,
        **kwargs,
    )


EMBEDDING_CACHES.register(
    DEFAULT_EMBEDDING_CACHE_PROVIDER,
    _build_no_embedding_cache,
)
EMBEDDING_CACHES.register(InMemoryCache.provider_name, _build_in_memory_embedding_cache)
EMBEDDING_CACHES.register(SqliteCache.provider_name, _build_sqlite_embedding_cache)

__all__ = [
    "DEFAULT_EMBEDDING_CACHE_PROVIDER",
    "InMemoryCache",
    "NoCache",
    "SqliteCache",
    "create_embedding_cache",
]
