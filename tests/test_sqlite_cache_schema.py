from __future__ import annotations

import asyncio
import sqlite3
import struct
from pathlib import Path

from rag_core.search.providers.chunk_context_cache import (
    ChunkContextKey,
    SqliteChunkContextCache,
)
from rag_core.search.providers.embedding_cache import (
    SqliteCache,
)
from rag_core.search.providers.embedding_cache_models import (
    EmbedCacheKey,
)


def test_sqlite_embedding_cache_resets_same_column_incompatible_table(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "embedding.sqlite"
    key = EmbedCacheKey(
        provider="provider",
        provider_config_fingerprint="provider-config",
        model="model",
        dimensions=1,
        input_type="document",
        normalization="none",
        processing_fingerprint="processing",
        content_sha256="content",
    )
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            "CREATE TABLE embedding_cache ("
            "key TEXT, vector BLOB NOT NULL, "
            "provider TEXT NOT NULL, provider_config_fingerprint TEXT NOT NULL, "
            "model TEXT NOT NULL, dimensions INTEGER NOT NULL, "
            "input_type TEXT NOT NULL, normalization TEXT NOT NULL, "
            "processing_fingerprint TEXT NOT NULL, ts INTEGER NOT NULL"
            ")"
        )
        connection.execute(
            "INSERT INTO embedding_cache "
            "(key, vector, provider, provider_config_fingerprint, model, dimensions, "
            "input_type, normalization, processing_fingerprint, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key.stringify(),
                struct.pack("f", 99.0),
                key.provider,
                key.provider_config_fingerprint,
                key.model,
                key.dimensions,
                key.input_type,
                key.normalization,
                key.processing_fingerprint,
                1,
            ),
        )
        connection.commit()
    finally:
        connection.close()

    cache = SqliteCache(db_path)
    try:
        asyncio.run(cache.put(key, [1.25]))
        assert asyncio.run(cache.get(key)) == [1.25]
    finally:
        cache.close()

    verification = sqlite3.connect(db_path)
    try:
        schema = _table_schema(verification, "embedding_cache")
        row_count = verification.execute(
            "SELECT COUNT(*) FROM embedding_cache"
        ).fetchone()[0]
    finally:
        verification.close()

    assert schema == SqliteCache._EXPECTED_SCHEMA
    assert row_count == 1


def test_sqlite_chunk_context_cache_resets_incompatible_table(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "chunk_context.sqlite"
    key = ChunkContextKey(
        contextualizer_id="ctx",
        document_sha256="doc",
        document_filename_sha256="filename",
        chunk_text_sha256="chunk",
        chunk_index=0,
        total_chunks=1,
        namespace="",
        corpus_id="",
        document_id="",
    )
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            "CREATE TABLE chunk_context_cache ("
            "key TEXT, context TEXT NOT NULL, "
            "contextualizer_id TEXT NOT NULL, ts INTEGER NOT NULL"
            ")"
        )
        connection.execute(
            "INSERT INTO chunk_context_cache "
            "(key, context, contextualizer_id, ts) VALUES (?, ?, ?, ?)",
            (key.stringify(), "stale-context", key.contextualizer_id, 1),
        )
        connection.commit()
    finally:
        connection.close()

    cache = SqliteChunkContextCache(db_path)
    try:
        asyncio.run(cache.put(key, "fresh-context"))
        assert asyncio.run(cache.get(key)) == "fresh-context"
    finally:
        cache.close()

    verification = sqlite3.connect(db_path)
    try:
        schema = _table_schema(verification, "chunk_context_cache")
        row_count = verification.execute(
            "SELECT COUNT(*) FROM chunk_context_cache"
        ).fetchone()[0]
    finally:
        verification.close()

    assert schema == SqliteChunkContextCache._EXPECTED_SCHEMA
    assert row_count == 1


def _table_schema(
    connection: sqlite3.Connection,
    table_name: str,
) -> frozenset[tuple[str, str, int, int]]:
    return frozenset(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    )
