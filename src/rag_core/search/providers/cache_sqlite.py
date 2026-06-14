from __future__ import annotations

import math
import sqlite3
import time
from typing import Final

_EVICTABLE_CACHE_TABLES: Final[frozenset[str]] = frozenset(
    {"embedding_cache", "chunk_context_cache"}
)


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


def _deleted_count(cursor: sqlite3.Cursor) -> int:
    rowcount = cursor.rowcount
    return int(rowcount) if rowcount is not None and rowcount >= 0 else 0


__all__ = [
    "evict_sqlite_cache_rows",
    "sqlite_table_schema",
    "validate_sqlite_cache_bounds",
]
