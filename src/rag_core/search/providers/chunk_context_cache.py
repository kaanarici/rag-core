from __future__ import annotations

import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

from rag_core.private_files import (
    harden_private_file,
    prepare_private_file_for_open,
)
from rag_core.search.providers.cache_provider_names import (
    IN_MEMORY_CACHE_PROVIDER,
    NO_CACHE_PROVIDER,
    SQLITE_CACHE_PROVIDER,
)
from rag_core.search.providers.cache_sqlite import sqlite_table_schema
from rag_core.search.providers.registry import CHUNK_CONTEXT_CACHES


@dataclass(frozen=True)
class ChunkContextKey:
    """Key for a chunk-context cache entry."""

    contextualizer_id: str
    document_sha256: str
    document_filename_sha256: str
    chunk_text_sha256: str
    chunk_index: int
    total_chunks: int

    def stringify(self) -> str:
        return (
            f"{self.contextualizer_id}|{self.document_sha256}|"
            f"{self.document_filename_sha256}|{self.chunk_text_sha256}|"
            f"{self.chunk_index}|{self.total_chunks}"
        )


@runtime_checkable
class ChunkContextCache(Protocol):
    """Look up and store contextualizer outputs."""

    async def get(self, key: ChunkContextKey) -> str | None: ...

    async def get_many(self, keys: list[ChunkContextKey]) -> dict[ChunkContextKey, str]: ...

    async def put(self, key: ChunkContextKey, context: str) -> None: ...

    async def put_many(self, items: dict[ChunkContextKey, str]) -> None: ...


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


class InMemoryChunkContextCache:
    """Dict-backed chunk-context cache."""

    provider_name = IN_MEMORY_CACHE_PROVIDER

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def get(self, key: ChunkContextKey) -> str | None:
        return self._store.get(key.stringify())

    async def get_many(self, keys: list[ChunkContextKey]) -> dict[ChunkContextKey, str]:
        return {
            key: self._store[cache_key]
            for key in keys
            if (cache_key := key.stringify()) in self._store
        }

    async def put(self, key: ChunkContextKey, context: str) -> None:
        self._store[key.stringify()] = context

    async def put_many(self, items: dict[ChunkContextKey, str]) -> None:
        for key, context in items.items():
            self._store[key.stringify()] = context


class SqliteChunkContextCache:
    """Disk-backed chunk-context cache."""

    provider_name = SQLITE_CACHE_PROVIDER

    _CREATE_SQL = (
        "CREATE TABLE IF NOT EXISTS chunk_context_cache ("
        "key TEXT PRIMARY KEY, context TEXT NOT NULL, "
        "contextualizer_id TEXT NOT NULL, ts INTEGER NOT NULL"
        ")"
    )
    _EXPECTED_SCHEMA: ClassVar[frozenset[tuple[str, str, int, int]]] = frozenset(
        {
            ("key", "TEXT", 0, 1),
            ("context", "TEXT", 1, 0),
            ("contextualizer_id", "TEXT", 1, 0),
            ("ts", "INTEGER", 1, 0),
        }
    )

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        self._connection: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None:
            path = Path(self._path)
            prepare_private_file_for_open(path, reject_symlink=True)
            connection = sqlite3.connect(self._path)
            harden_private_file(path)
            self._ensure_schema(connection)
            connection.commit()
            self._connection = connection
        return self._connection

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
        schema = sqlite_table_schema(connection, "chunk_context_cache")
        if schema and schema != self._EXPECTED_SCHEMA:
            connection.execute("DROP TABLE chunk_context_cache")
        connection.execute(self._CREATE_SQL)

    async def get(self, key: ChunkContextKey) -> str | None:
        connection = self._connect()
        cursor = connection.execute(
            "SELECT context FROM chunk_context_cache WHERE key = ?",
            (key.stringify(),),
        )
        row = cursor.fetchone()
        return None if row is None else str(row[0])

    async def get_many(self, keys: list[ChunkContextKey]) -> dict[ChunkContextKey, str]:
        stringified_keys = [key.stringify() for key in keys]
        contexts: dict[str, str] = {}
        connection = self._connect()
        for key_batch in _chunks(list(dict.fromkeys(stringified_keys)), 900):
            placeholders = ",".join("?" for _ in key_batch)
            cursor = connection.execute(
                f"SELECT key, context FROM chunk_context_cache WHERE key IN ({placeholders})",
                key_batch,
            )
            contexts.update((str(row[0]), str(row[1])) for row in cursor.fetchall())
        return {
            key: contexts[cache_key]
            for key in keys
            if (cache_key := key.stringify()) in contexts
        }

    async def put(self, key: ChunkContextKey, context: str) -> None:
        connection = self._connect()
        connection.execute(
            "INSERT OR REPLACE INTO chunk_context_cache "
            "(key, context, contextualizer_id, ts) VALUES (?, ?, ?, ?)",
            (
                key.stringify(),
                context,
                key.contextualizer_id,
                int(time.time()),
            ),
        )
        connection.commit()

    async def put_many(self, items: dict[ChunkContextKey, str]) -> None:
        if not items:
            return
        timestamp = int(time.time())
        connection = self._connect()
        connection.executemany(
            "INSERT OR REPLACE INTO chunk_context_cache "
            "(key, context, contextualizer_id, ts) VALUES (?, ?, ?, ?)",
            [
                (key.stringify(), context, key.contextualizer_id, timestamp)
                for key, context in items.items()
            ],
        )
        connection.commit()

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None


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
    "SqliteChunkContextCache",
    "create_chunk_context_cache",
]
