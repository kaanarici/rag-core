from __future__ import annotations

import asyncio
import math
import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, TypeVar
from typing import Final

from rag_core.search.providers.cache_sqlite_connection import (
    SqliteCacheConnectionPool,
    bind_running_task_for_pool,
    harden_sqlite_sidecar_files,
)

NO_CACHE_PROVIDER = "none"
IN_MEMORY_CACHE_PROVIDER = "in_memory"
SQLITE_CACHE_PROVIDER = "sqlite"
CACHE_PROVIDER_ORDER = (
    NO_CACHE_PROVIDER,
    IN_MEMORY_CACHE_PROVIDER,
    SQLITE_CACHE_PROVIDER,
)

_EVICTABLE_CACHE_TABLES: Final[frozenset[str]] = frozenset(
    {"embedding_cache", "chunk_context_cache"}
)
_RunResult = TypeVar("_RunResult")


@dataclass(frozen=True)
class SqliteCacheSchema:
    table_name: str
    create_table_sql: str
    expected_schema: frozenset[tuple[str, str, int, int]]
    create_scope_index_sql: str
    create_eviction_index_sql: str


class SqliteCacheBase:
    """Shared async SQLite lifecycle for cache providers.

    Each concrete cache owns its key/value schema and row encoding. This base
    owns the common execution model: validate bounds, lease a connection bound
    to the originating asyncio task, prepare schema once per connection, run
    blocking SQL in a worker thread, and close the per-task pool.
    """

    _SCHEMA: ClassVar[SqliteCacheSchema]

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
        bind_running_task_for_pool(self._pool)

        def _bound() -> _RunResult:
            with self._pool.connection() as connection:
                self._prepare_connection(connection)
                return func(connection, *args, **kwargs)

        return await asyncio.to_thread(_bound)

    def _prepare_connection(self, connection: sqlite3.Connection) -> None:
        connection_id = id(connection)
        if connection_id in self._schema_ready:
            return
        schema = self._SCHEMA
        current = sqlite_table_schema(connection, schema.table_name)
        if current and current != schema.expected_schema:
            connection.execute(f"DROP TABLE {schema.table_name}")
        connection.execute(schema.create_table_sql)
        connection.execute(schema.create_scope_index_sql)
        if self._eviction_enabled:
            connection.execute(schema.create_eviction_index_sql)
            evict_sqlite_cache_rows(
                connection,
                table_name=schema.table_name,
                max_age_seconds=self._max_age_seconds,
                max_entries=self._max_entries,
            )
        connection.commit()
        harden_sqlite_sidecar_files(Path(self._path))
        self._schema_ready.add(connection_id)

    def close(self) -> None:
        self._pool.close()
        self._schema_ready.clear()


def sqlite_table_schema(
    connection: sqlite3.Connection,
    table_name: str,
) -> frozenset[tuple[str, str, int, int]]:
    return frozenset(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    )


def validate_sqlite_cache_bounds(
    *,
    max_age_seconds: float | None,
    max_entries: int | None,
) -> None:
    if max_age_seconds is not None and (
        max_age_seconds <= 0 or not math.isfinite(max_age_seconds)
    ):
        raise ValueError("max_age_seconds must be positive or None")
    if max_entries is not None and max_entries <= 0:
        raise ValueError("max_entries must be positive or None")


def evict_sqlite_cache_rows(
    connection: sqlite3.Connection,
    *,
    table_name: str,
    max_age_seconds: float | None,
    max_entries: int | None,
    now: float | None = None,
) -> int:
    validate_sqlite_cache_bounds(
        max_age_seconds=max_age_seconds,
        max_entries=max_entries,
    )
    if table_name not in _EVICTABLE_CACHE_TABLES:
        raise ValueError(f"unsupported sqlite cache eviction table: {table_name!r}")

    deleted = 0
    if max_age_seconds is not None:
        cursor = connection.execute(
            f"DELETE FROM {table_name} WHERE ts < ?",
            ((time.time() if now is None else now) - max_age_seconds,),
        )
        deleted += _deleted_count(cursor)
    if max_entries is not None:
        cursor = connection.execute(
            f"DELETE FROM {table_name} "
            "WHERE key IN ("
            f"SELECT key FROM {table_name} "
            "ORDER BY ts DESC, key DESC LIMIT -1 OFFSET ?"
            ")",
            (max_entries,),
        )
        deleted += _deleted_count(cursor)
    return deleted


def fetch_one(
    connection: sqlite3.Connection,
    sql: str,
    params: tuple[object, ...],
) -> tuple[object, ...] | None:
    row = connection.execute(sql, params).fetchone()
    return None if row is None else tuple(row)


def fetch_keyed_text_values(
    connection: sqlite3.Connection,
    *,
    table_name: str,
    value_column: str,
    cache_keys: Sequence[str],
) -> dict[str, str]:
    if table_name not in _EVICTABLE_CACHE_TABLES:
        raise ValueError(f"unsupported sqlite cache table: {table_name!r}")
    values: dict[str, str] = {}
    for key_batch in chunked(cache_keys, 900):
        placeholders = ",".join("?" for _ in key_batch)
        cursor = connection.execute(
            f"SELECT key, {value_column} FROM {table_name} WHERE key IN ({placeholders})",
            list(key_batch),
        )
        values.update((str(row[0]), str(row[1])) for row in cursor.fetchall())
    return values


def execute_write_with_eviction(
    connection: sqlite3.Connection,
    *,
    sql: str,
    params: tuple[object, ...],
    table_name: str,
    max_age_seconds: float | None = None,
    max_entries: int | None = None,
    now: float | None = None,
) -> None:
    connection.execute(sql, params)
    evict_after_write(
        connection,
        table_name=table_name,
        max_age_seconds=max_age_seconds,
        max_entries=max_entries,
        now=now,
    )
    connection.commit()


def executemany_with_eviction(
    connection: sqlite3.Connection,
    *,
    sql: str,
    rows: Sequence[tuple[object, ...]],
    table_name: str,
    max_age_seconds: float | None = None,
    max_entries: int | None = None,
    now: float | None = None,
) -> None:
    if not rows:
        return
    connection.executemany(sql, rows)
    evict_after_write(
        connection,
        table_name=table_name,
        max_age_seconds=max_age_seconds,
        max_entries=max_entries,
        now=now,
    )
    connection.commit()


def evict_after_write(
    connection: sqlite3.Connection,
    *,
    table_name: str,
    max_age_seconds: float | None,
    max_entries: int | None,
    now: float | None = None,
) -> None:
    if max_age_seconds is None and max_entries is None:
        return
    evict_sqlite_cache_rows(
        connection,
        table_name=table_name,
        max_age_seconds=max_age_seconds,
        max_entries=max_entries,
        now=now,
    )


def delete_cache_rows_by_scope(
    connection: sqlite3.Connection,
    *,
    table_name: str,
    namespace: str,
    collection: str,
    document_id: str,
) -> int:
    if table_name not in _EVICTABLE_CACHE_TABLES:
        raise ValueError(f"unsupported sqlite cache table: {table_name!r}")
    cursor = connection.execute(
        f"DELETE FROM {table_name} "
        "WHERE namespace = ? AND collection = ? AND document_id = ?",
        (namespace, collection, document_id),
    )
    connection.commit()
    return _deleted_count(cursor)


def chunked(values: Sequence[str], size: int) -> list[Sequence[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _deleted_count(cursor: sqlite3.Cursor) -> int:
    rowcount = cursor.rowcount
    return int(rowcount) if rowcount is not None and rowcount >= 0 else 0


__all__ = [
    "CACHE_PROVIDER_ORDER",
    "IN_MEMORY_CACHE_PROVIDER",
    "NO_CACHE_PROVIDER",
    "SQLITE_CACHE_PROVIDER",
    "SqliteCacheBase",
    "SqliteCacheSchema",
    "chunked",
    "delete_cache_rows_by_scope",
    "evict_after_write",
    "evict_sqlite_cache_rows",
    "execute_write_with_eviction",
    "executemany_with_eviction",
    "fetch_keyed_text_values",
    "fetch_one",
    "sqlite_table_schema",
    "validate_sqlite_cache_bounds",
]
