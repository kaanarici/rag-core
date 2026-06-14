"""SQL-side helpers for :class:`SqliteChunkContextCache`.

Extracted from ``chunk_context_cache.py`` so the cache module stays focused
on the public ``ChunkContextCache`` protocol and the per-task connection
pool wiring. Each helper takes an already-acquired ``sqlite3.Connection``
and is dispatched from the cache via ``asyncio.to_thread``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

from rag_core.search.providers.cache_sqlite import (
    evict_sqlite_cache_rows,
)

CREATE_CHUNK_CONTEXT_CACHE_TS_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_chunk_context_cache_ts "
    "ON chunk_context_cache (ts, key)"
)


def ensure_chunk_context_cache_eviction_index(
    connection: sqlite3.Connection,
) -> None:
    connection.execute(CREATE_CHUNK_CONTEXT_CACHE_TS_INDEX_SQL)


def fetchone(
    connection: sqlite3.Connection,
    sql: str,
    params: tuple[object, ...],
) -> tuple[object, ...] | None:
    cursor = connection.execute(sql, params)
    row = cursor.fetchone()
    if row is None:
        return None
    return tuple(row)


def fetch_chunk_context_batch(
    connection: sqlite3.Connection,
    cache_keys: list[str],
) -> dict[str, str]:
    contexts: dict[str, str] = {}
    for key_batch in _chunks(cache_keys, 900):
        placeholders = ",".join("?" for _ in key_batch)
        cursor = connection.execute(
            f"SELECT key, context FROM chunk_context_cache WHERE key IN ({placeholders})",
            list(key_batch),
        )
        contexts.update((str(row[0]), str(row[1])) for row in cursor.fetchall())
    return contexts


def execute_put(
    connection: sqlite3.Connection,
    sql: str,
    params: tuple[object, ...],
    *,
    max_age_seconds: float | None = None,
    max_entries: int | None = None,
    now: float | None = None,
) -> None:
    connection.execute(sql, params)
    if max_age_seconds is not None or max_entries is not None:
        evict_sqlite_cache_rows(
            connection,
            table_name="chunk_context_cache",
            max_age_seconds=max_age_seconds,
            max_entries=max_entries,
            now=now,
        )
    connection.commit()


def executemany_put(
    connection: sqlite3.Connection,
    sql: str,
    rows: list[tuple[object, ...]],
    *,
    max_age_seconds: float | None = None,
    max_entries: int | None = None,
    now: float | None = None,
) -> None:
    connection.executemany(sql, rows)
    if max_age_seconds is not None or max_entries is not None:
        evict_sqlite_cache_rows(
            connection,
            table_name="chunk_context_cache",
            max_age_seconds=max_age_seconds,
            max_entries=max_entries,
            now=now,
        )
    connection.commit()


def execute_delete_by_scope(
    connection: sqlite3.Connection,
    namespace: str,
    corpus_id: str,
    document_id: str,
) -> int:
    cursor = connection.execute(
        "DELETE FROM chunk_context_cache "
        "WHERE namespace = ? AND corpus_id = ? AND document_id = ?",
        (namespace, corpus_id, document_id),
    )
    connection.commit()
    deleted = cursor.rowcount
    return int(deleted) if deleted is not None and deleted >= 0 else 0


def _chunks(values: Sequence[str], size: int) -> list[Sequence[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


__all__ = [
    "ensure_chunk_context_cache_eviction_index",
    "execute_delete_by_scope",
    "execute_put",
    "executemany_put",
    "fetch_chunk_context_batch",
    "fetchone",
]
