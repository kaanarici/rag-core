from __future__ import annotations

import sqlite3


def sqlite_table_schema(
    connection: sqlite3.Connection,
    table_name: str,
) -> frozenset[tuple[str, str, int, int]]:
    return frozenset(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    )
