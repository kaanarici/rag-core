from __future__ import annotations

import sqlite3
import struct
import time
from collections.abc import Sequence
from dataclasses import dataclass

from rag_core.search.providers.cache_sqlite import sqlite_table_schema

CREATE_EMBEDDING_CACHE_SQL = (
    "CREATE TABLE IF NOT EXISTS embedding_cache ("
    "key TEXT PRIMARY KEY, vector BLOB NOT NULL, "
    "provider TEXT NOT NULL, provider_config_fingerprint TEXT NOT NULL, "
    "model TEXT NOT NULL, dimensions INTEGER NOT NULL, "
    "input_type TEXT NOT NULL, normalization TEXT NOT NULL, "
    "processing_fingerprint TEXT NOT NULL, ts INTEGER NOT NULL"
    ")"
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
        ("ts", "INTEGER", 1, 0),
    }
)


@dataclass(frozen=True)
class EmbeddingCacheWrite:
    cache_key: str
    vector: list[float]
    provider: str
    provider_config_fingerprint: str
    model: str
    dimensions: int
    input_type: str
    normalization: str
    processing_fingerprint: str


def ensure_embedding_cache_schema(connection: sqlite3.Connection) -> None:
    schema = sqlite_table_schema(connection, "embedding_cache")
    if schema and schema != EXPECTED_EMBEDDING_CACHE_SCHEMA:
        connection.execute("DROP TABLE embedding_cache")
    connection.execute(CREATE_EMBEDDING_CACHE_SQL)


def read_embedding_cache_vector(
    connection: sqlite3.Connection,
    *,
    cache_key: str,
) -> list[float] | None:
    cursor = connection.execute(
        "SELECT vector FROM embedding_cache WHERE key = ?",
        (cache_key,),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    try:
        return _decode_vector(row[0])
    except (struct.error, TypeError):
        connection.execute(
            "DELETE FROM embedding_cache WHERE key = ?",
            (cache_key,),
        )
        connection.commit()
        return None


def read_embedding_cache_vectors(
    connection: sqlite3.Connection,
    *,
    cache_keys: Sequence[str],
) -> dict[str, list[float]]:
    unique_keys = list(dict.fromkeys(cache_keys))
    if not unique_keys:
        return {}

    vectors: dict[str, list[float]] = {}
    corrupt_keys: list[str] = []
    for key_batch in _chunks(unique_keys, 900):
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
        for key_batch in _chunks(corrupt_keys, 900):
            placeholders = ",".join("?" for _ in key_batch)
            connection.execute(
                f"DELETE FROM embedding_cache WHERE key IN ({placeholders})",
                key_batch,
            )
        connection.commit()
    return vectors


def write_embedding_cache_vector(
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
) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO embedding_cache "
        "(key, vector, provider, provider_config_fingerprint, model, dimensions, input_type, "
        "normalization, processing_fingerprint, ts) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
            int(time.time()),
        ),
    )
    connection.commit()


def write_embedding_cache_vectors(
    connection: sqlite3.Connection,
    *,
    entries: Sequence[EmbeddingCacheWrite],
) -> None:
    if not entries:
        return
    timestamp = int(time.time())
    connection.executemany(
        "INSERT OR REPLACE INTO embedding_cache "
        "(key, vector, provider, provider_config_fingerprint, model, dimensions, input_type, "
        "normalization, processing_fingerprint, ts) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                timestamp,
            )
            for entry in entries
        ],
    )
    connection.commit()


def _chunks(values: Sequence[str], size: int) -> list[Sequence[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _encode_vector(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def _decode_vector(blob: bytes) -> list[float]:
    if not blob:
        return []
    if len(blob) % 4 != 0:
        raise struct.error("embedding cache vector blob length must be a multiple of 4")
    count = len(blob) // 4
    return list(struct.unpack(f"{count}f", blob))
