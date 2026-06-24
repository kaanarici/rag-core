"""Concrete embedding cache providers and cache factory registrations."""

from __future__ import annotations

import sqlite3
import struct
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from rag_core.search.providers.cache_sqlite import (
    IN_MEMORY_CACHE_PROVIDER,
    NO_CACHE_PROVIDER,
    SQLITE_CACHE_PROVIDER,
    SqliteCacheBase,
    SqliteCacheSchema,
    chunked,
    delete_cache_rows_by_scope,
    evict_after_write,
)
from rag_core.search.providers.registry import (
    EMBEDDING_CACHES,
)
from rag_core.search.providers.embedding_cache_models import (
    EmbedCacheKey as _EmbedCacheKey,
    EmbeddingCache as _EmbeddingCache,
)

CREATE_EMBEDDING_CACHE_SQL = (
    "CREATE TABLE IF NOT EXISTS embedding_cache ("
    "key TEXT PRIMARY KEY, vector BLOB NOT NULL, "
    "provider TEXT NOT NULL, provider_config_fingerprint TEXT NOT NULL, "
    "model TEXT NOT NULL, dimensions INTEGER NOT NULL, "
    "input_type TEXT NOT NULL, normalization TEXT NOT NULL, "
    "processing_fingerprint TEXT NOT NULL, "
    "namespace TEXT NOT NULL DEFAULT '', "
    "collection TEXT NOT NULL DEFAULT '', "
    "document_id TEXT NOT NULL DEFAULT '', "
    "ts INTEGER NOT NULL"
    ")"
)
CREATE_EMBEDDING_CACHE_SCOPE_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_embedding_cache_scope "
    "ON embedding_cache (namespace, collection, document_id)"
)
CREATE_EMBEDDING_CACHE_TS_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_embedding_cache_ts "
    "ON embedding_cache (ts, key)"
)
EXPECTED_EMBEDDING_CACHE_SCHEMA: frozenset[tuple[str, str, int, int]] = frozenset(
    {
        ("key", "TEXT", 0, 1),
        ("vector", "BLOB", 1, 0),
        ("provider", "TEXT", 1, 0),
        ("provider_config_fingerprint", "TEXT", 1, 0),
        ("model", "TEXT", 1, 0),
        ("dimensions", "INTEGER", 1, 0),
        ("input_type", "TEXT", 1, 0),
        ("normalization", "TEXT", 1, 0),
        ("processing_fingerprint", "TEXT", 1, 0),
        ("namespace", "TEXT", 1, 0),
        ("collection", "TEXT", 1, 0),
        ("document_id", "TEXT", 1, 0),
        ("ts", "INTEGER", 1, 0),
    }
)
_EMBEDDING_SCHEMA = SqliteCacheSchema(
    table_name="embedding_cache",
    create_table_sql=CREATE_EMBEDDING_CACHE_SQL,
    expected_schema=EXPECTED_EMBEDDING_CACHE_SCHEMA,
    create_scope_index_sql=CREATE_EMBEDDING_CACHE_SCOPE_INDEX_SQL,
    create_eviction_index_sql=CREATE_EMBEDDING_CACHE_TS_INDEX_SQL,
)


class NoCache:
    """Default cache that always misses."""

    provider_name = NO_CACHE_PROVIDER

    async def get(self, key: _EmbedCacheKey) -> list[float] | None:
        return None

    async def put(self, key: _EmbedCacheKey, vector: list[float]) -> None:
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


class InMemoryCache:
    """LRU dict-backed embedding cache with scope-keyed reverse index."""

    provider_name = IN_MEMORY_CACHE_PROVIDER

    def __init__(self, *, max_entries: int | None = None) -> None:
        if max_entries is not None and max_entries <= 0:
            raise ValueError("max_entries must be positive or None")
        self._max_entries = max_entries
        self._store: OrderedDict[str, list[float]] = OrderedDict()
        self._scope_index: dict[tuple[str, str, str], set[str]] = {}

    async def get(self, key: _EmbedCacheKey) -> list[float] | None:
        cache_key = key.stringify()
        if cache_key not in self._store:
            return None
        value = self._store.pop(cache_key)
        self._store[cache_key] = value
        return list(value)

    async def put(self, key: _EmbedCacheKey, vector: list[float]) -> None:
        cache_key = key.stringify()
        if cache_key in self._store:
            self._store.pop(cache_key)
        self._store[cache_key] = list(vector)
        scope = (key.namespace, key.collection, key.document_id)
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


@dataclass(frozen=True)
class _EmbeddingCacheWrite:
    cache_key: str
    vector: list[float]
    provider: str
    provider_config_fingerprint: str
    model: str
    dimensions: int
    input_type: str
    normalization: str
    processing_fingerprint: str
    namespace: str = ""
    collection: str = ""
    document_id: str = ""


class SqliteCache(SqliteCacheBase):
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
    _SCHEMA: ClassVar[SqliteCacheSchema] = _EMBEDDING_SCHEMA

    _EXPECTED_SCHEMA: ClassVar[frozenset[tuple[str, str, int, int]]] = frozenset(
        EXPECTED_EMBEDDING_CACHE_SCHEMA
    )

    async def get(self, key: _EmbedCacheKey) -> list[float] | None:
        result: list[float] | None = await self._run(
            _read_embedding_cache_vector,
            cache_key=key.stringify(),
        )
        return result

    async def get_many(
        self,
        keys: list[_EmbedCacheKey],
    ) -> dict[_EmbedCacheKey, list[float]]:
        vectors = await self._run(
            _read_embedding_cache_vectors,
            cache_keys=[key.stringify() for key in keys],
        )
        return {
            key: vectors[cache_key]
            for key in keys
            if (cache_key := key.stringify()) in vectors
        }

    async def put(self, key: _EmbedCacheKey, vector: list[float]) -> None:
        await self._run(
            _write_embedding_cache_vector,
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
            collection=key.collection,
            document_id=key.document_id,
            max_age_seconds=self._max_age_seconds,
            max_entries=self._max_entries,
        )

    async def put_many(
        self,
        items: dict[_EmbedCacheKey, list[float]],
    ) -> None:
        entries = [
            _EmbeddingCacheWrite(
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
                collection=key.collection,
                document_id=key.document_id,
            )
            for key, vector in items.items()
        ]
        await self._run(
            _write_embedding_cache_vectors,
            entries=entries,
            max_age_seconds=self._max_age_seconds,
            max_entries=self._max_entries,
        )

    async def delete_by_document_scope(
        self,
        *,
        namespace: str,
        collection: str,
        document_id: str,
    ) -> int:
        """Purge every cached vector tagged to ``(namespace, collection, document_id)``.

        Returns the number of rows removed so the right-to-forget facade can
        record an honest ``DeleteDocumentResult.embedding_cache_purged``.
        """
        deleted: int = await self._run(
            delete_cache_rows_by_scope,
            table_name="embedding_cache",
            namespace=namespace,
            collection=collection,
            document_id=document_id,
        )
        return deleted


def _read_embedding_cache_vector(
    connection: sqlite3.Connection,
    *,
    cache_key: str,
) -> list[float] | None:
    row = connection.execute(
        "SELECT vector FROM embedding_cache WHERE key = ?",
        (cache_key,),
    ).fetchone()
    if row is None:
        return None
    try:
        return _decode_vector(row[0])
    except (struct.error, TypeError):
        connection.execute("DELETE FROM embedding_cache WHERE key = ?", (cache_key,))
        connection.commit()
        return None


def _read_embedding_cache_vectors(
    connection: sqlite3.Connection,
    *,
    cache_keys: Sequence[str],
) -> dict[str, list[float]]:
    unique_keys = list(dict.fromkeys(cache_keys))
    if not unique_keys:
        return {}

    vectors: dict[str, list[float]] = {}
    corrupt_keys: list[str] = []
    for key_batch in chunked(unique_keys, 900):
        placeholders = ",".join("?" for _ in key_batch)
        cursor = connection.execute(
            f"SELECT key, vector FROM embedding_cache WHERE key IN ({placeholders})",
            key_batch,
        )
        for row_key, blob in cursor.fetchall():
            cache_key = str(row_key)
            try:
                vectors[cache_key] = _decode_vector(blob)
            except (struct.error, TypeError):
                corrupt_keys.append(cache_key)

    if corrupt_keys:
        for key_batch in chunked(corrupt_keys, 900):
            placeholders = ",".join("?" for _ in key_batch)
            connection.execute(
                f"DELETE FROM embedding_cache WHERE key IN ({placeholders})",
                key_batch,
            )
        connection.commit()
    return vectors


def _write_embedding_cache_vector(
    connection: sqlite3.Connection,
    *,
    cache_key: str,
    vector: list[float],
    provider: str,
    provider_config_fingerprint: str,
    model: str,
    dimensions: int,
    input_type: str,
    normalization: str,
    processing_fingerprint: str,
    namespace: str = "",
    collection: str = "",
    document_id: str = "",
    max_age_seconds: float | None = None,
    max_entries: int | None = None,
) -> None:
    timestamp = int(time.time())
    connection.execute(
        "INSERT OR REPLACE INTO embedding_cache "
        "(key, vector, provider, provider_config_fingerprint, model, dimensions, input_type, "
        "normalization, processing_fingerprint, namespace, collection, document_id, ts) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            cache_key,
            _encode_vector(vector),
            provider,
            provider_config_fingerprint,
            model,
            dimensions,
            input_type,
            normalization,
            processing_fingerprint,
            namespace,
            collection,
            document_id,
            timestamp,
        ),
    )
    evict_after_write(
        connection,
        table_name="embedding_cache",
        max_age_seconds=max_age_seconds,
        max_entries=max_entries,
        now=timestamp,
    )
    connection.commit()


def _write_embedding_cache_vectors(
    connection: sqlite3.Connection,
    *,
    entries: Sequence[_EmbeddingCacheWrite],
    max_age_seconds: float | None = None,
    max_entries: int | None = None,
) -> None:
    if not entries:
        return
    timestamp = int(time.time())
    connection.executemany(
        "INSERT OR REPLACE INTO embedding_cache "
        "(key, vector, provider, provider_config_fingerprint, model, dimensions, input_type, "
        "normalization, processing_fingerprint, namespace, collection, document_id, ts) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                entry.cache_key,
                _encode_vector(entry.vector),
                entry.provider,
                entry.provider_config_fingerprint,
                entry.model,
                entry.dimensions,
                entry.input_type,
                entry.normalization,
                entry.processing_fingerprint,
                entry.namespace,
                entry.collection,
                entry.document_id,
                timestamp,
            )
            for entry in entries
        ],
    )
    evict_after_write(
        connection,
        table_name="embedding_cache",
        max_age_seconds=max_age_seconds,
        max_entries=max_entries,
        now=timestamp,
    )
    connection.commit()


def _encode_vector(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def _decode_vector(blob: bytes) -> list[float]:
    if not blob:
        return []
    if len(blob) % 4 != 0:
        raise struct.error("embedding cache vector blob length must be a multiple of 4")
    count = len(blob) // 4
    return list(struct.unpack(f"{count}f", blob))


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
