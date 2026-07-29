from __future__ import annotations

import asyncio
import os
import sqlite3
from pathlib import Path

import pytest

from rag_core.documents.contextualizer import ChunkContextRequest
from rag_core.documents.contextualizer_adapters import CachingContextualizer
from rag_core.search.providers.chunk_context_cache import (
    ChunkContextKey,
    InMemoryChunkContextCache,
    NoChunkContextCache,
    SqliteChunkContextCache,
)
from rag_core.search.providers.embedding_cache_models import sha256_text


def _request(
    *,
    chunk_index: int = 0,
    total_chunks: int = 2,
    chunk_text: str = "alpha",
    document_markdown: str = "doc body",
) -> ChunkContextRequest:
    return ChunkContextRequest(
        document_markdown=document_markdown,
        document_filename="doc.md",
        chunk_text=chunk_text,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
    )


def _key(
    *,
    contextualizer_id: str = "x",
    document_sha256: str = "d",
    document_filename_sha256: str = "f",
    chunk_text_sha256: str = "c",
    chunk_index: int = 0,
    total_chunks: int = 2,
) -> ChunkContextKey:
    return ChunkContextKey(
        contextualizer_id=contextualizer_id,
        document_sha256=document_sha256,
        document_filename_sha256=document_filename_sha256,
        chunk_text_sha256=chunk_text_sha256,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        namespace="",
        collection="",
        document_id="",
    )


def _set_chunk_context_cache_ts(
    db_path: Path,
    key: ChunkContextKey,
    ts: int,
) -> None:
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            "UPDATE chunk_context_cache SET ts = ? WHERE key = ?",
            (ts, key.stringify()),
        )
        connection.commit()
    finally:
        connection.close()


class _RecordingContextualizer:
    contextualizer_id = "test:recording"

    def __init__(self) -> None:
        self.calls: list[int] = []

    async def contextualize(self, request: ChunkContextRequest) -> str:
        self.calls.append(request.chunk_index)
        return f"context-{request.chunk_index}"


def test_chunk_context_key_stringify_is_stable() -> None:
    key = ChunkContextKey(
        contextualizer_id="anthropic:claude",
        document_sha256="hash",
        document_filename_sha256="filename-hash",
        chunk_text_sha256="chunk-hash",
        chunk_index=4,
        total_chunks=9,
        namespace="",
        collection="",
        document_id="",
    )
    # Direct prepare keys pass explicit empty scope; ingest keys populate the
    # trailing segments so scoped delete can purge them.
    assert (
        key.stringify()
        == "anthropic:claude|hash|filename-hash|chunk-hash|4|9|||"
    )


def test_chunk_context_key_stringify_carries_scope() -> None:
    key = ChunkContextKey(
        contextualizer_id="anthropic:claude",
        document_sha256="hash",
        document_filename_sha256="filename-hash",
        chunk_text_sha256="chunk-hash",
        chunk_index=0,
        total_chunks=1,
        namespace="workspace-alpha",
        collection="restricted",
        document_id="doc-007",
    )
    assert key.stringify().endswith("|workspace-alpha|restricted|doc-007")


def test_in_memory_chunk_context_cache_delete_by_document_scope() -> None:
    cache = InMemoryChunkContextCache()
    scoped = _key(chunk_index=0)
    # Build a scoped variant inline so the helper-built default scope keys
    # stay legacy-shaped for the existing tests.
    scoped_a = ChunkContextKey(
        contextualizer_id="x",
        document_sha256="d",
        document_filename_sha256="f",
        chunk_text_sha256="c",
        chunk_index=0,
        total_chunks=2,
        namespace="ns",
        collection="cor",
        document_id="doc-A",
    )
    scoped_b = ChunkContextKey(
        contextualizer_id="x",
        document_sha256="d",
        document_filename_sha256="f",
        chunk_text_sha256="c",
        chunk_index=0,
        total_chunks=2,
        namespace="ns",
        collection="cor",
        document_id="doc-B",
    )
    asyncio.run(cache.put(scoped_a, "context-a"))
    asyncio.run(cache.put(scoped_b, "context-b"))
    asyncio.run(cache.put(scoped, "legacy"))

    removed = asyncio.run(
        cache.delete_by_document_scope(
            namespace="ns",
            collection="cor",
            document_id="doc-A",
        )
    )

    assert removed == 1
    assert asyncio.run(cache.get(scoped_a)) is None
    # doc-B and the legacy un-scoped entry survive the scoped purge.
    assert asyncio.run(cache.get(scoped_b)) == "context-b"
    assert asyncio.run(cache.get(scoped)) == "legacy"


def test_no_chunk_context_cache_always_misses() -> None:
    cache = NoChunkContextCache()
    assert asyncio.run(cache.get(_key())) is None
    asyncio.run(cache.put(_key(), "context"))
    assert asyncio.run(cache.get(_key())) is None


def test_in_memory_chunk_context_cache_round_trips() -> None:
    cache = InMemoryChunkContextCache()
    key = _key(chunk_index=2)
    asyncio.run(cache.put(key, "stored"))
    assert asyncio.run(cache.get(key)) == "stored"


def test_in_memory_chunk_context_cache_batch_round_trips() -> None:
    cache = InMemoryChunkContextCache()
    key_a = _key(chunk_index=0)
    key_b = _key(chunk_index=1)
    key_missing = _key(chunk_index=2)

    asyncio.run(cache.put_many({key_a: "a", key_b: "b"}))

    assert asyncio.run(cache.get_many([key_a, key_b, key_missing])) == {
        key_a: "a",
        key_b: "b",
    }


@pytest.mark.parametrize(
    "stored, probe",
    [
        ({"chunk_index": 0}, {"chunk_index": 1}),
        ({"contextualizer_id": "a"}, {"contextualizer_id": "b"}),
        ({"document_sha256": "d1"}, {"document_sha256": "d2"}),
        ({"document_filename_sha256": "f1"}, {"document_filename_sha256": "f2"}),
    ],
    ids=["chunk-index", "contextualizer-id", "document-sha256", "document-filename-sha256"],
)
def test_in_memory_chunk_context_cache_isolates_keys_per_field(
    stored: dict[str, object], probe: dict[str, object]
) -> None:
    cache = InMemoryChunkContextCache()
    asyncio.run(cache.put(_key(**stored), "stored"))  # type: ignore[arg-type]
    assert asyncio.run(cache.get(_key(**probe))) is None  # type: ignore[arg-type]


def test_sqlite_chunk_context_cache_round_trips_after_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "chunk_ctx.sqlite"
    cache = SqliteChunkContextCache(db_path)
    key = _key(chunk_index=1)
    asyncio.run(cache.put(key, "persistent"))
    cache.close()

    reopened = SqliteChunkContextCache(db_path)
    try:
        result = asyncio.run(reopened.get(key))
    finally:
        reopened.close()

    assert result == "persistent"


def test_sqlite_chunk_context_cache_batch_get_put_round_trips(tmp_path: Path) -> None:
    db_path = tmp_path / "chunk_ctx.sqlite"
    cache = SqliteChunkContextCache(db_path)
    key_a = _key(chunk_index=0)
    key_b = _key(chunk_index=1)
    key_missing = _key(chunk_index=2)
    try:
        asyncio.run(cache.put_many({key_a: "a", key_b: "b"}))
        result = asyncio.run(cache.get_many([key_a, key_b, key_missing]))
    finally:
        cache.close()

    assert result == {
        key_a: "a",
        key_b: "b",
    }


def test_sqlite_chunk_context_cache_creates_private_db_file_and_parent(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "cache" / "chunk_ctx.sqlite"
    cache = SqliteChunkContextCache(db_path)
    try:
        asyncio.run(cache.put(_key(chunk_index=1), "persistent"))
    finally:
        cache.close()

    if os.name != "nt":
        assert db_path.stat().st_mode & 0o777 == 0o600
        assert (tmp_path / "cache").stat().st_mode & 0o777 == 0o700


def test_sqlite_chunk_context_cache_rejects_symlink_db_path(tmp_path: Path) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlink support unavailable")
    target = tmp_path / "target.sqlite"
    db_path = tmp_path / "chunk_ctx.sqlite"
    db_path.symlink_to(target)
    cache = SqliteChunkContextCache(db_path)

    with pytest.raises(ValueError, match="path must not be a symlink"):
        asyncio.run(cache.put(_key(chunk_index=1), "private"))

    assert not target.exists()


@pytest.mark.parametrize("max_age_seconds", [0.0, -1.0, float("nan")])
def test_sqlite_chunk_context_cache_rejects_invalid_max_age_seconds(
    tmp_path: Path,
    max_age_seconds: float,
) -> None:
    with pytest.raises(ValueError, match="max_age_seconds"):
        SqliteChunkContextCache(
            tmp_path / "chunk_ctx.sqlite",
            max_age_seconds=max_age_seconds,
        )


@pytest.mark.parametrize("max_entries", [0, -1])
def test_sqlite_chunk_context_cache_rejects_invalid_max_entries(
    tmp_path: Path,
    max_entries: int,
) -> None:
    with pytest.raises(ValueError, match="max_entries"):
        SqliteChunkContextCache(tmp_path / "chunk_ctx.sqlite", max_entries=max_entries)


def test_sqlite_chunk_context_cache_unbounded_get_put_skip_eviction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail_eviction(*args: object, **kwargs: object) -> int:
        raise AssertionError("unbounded sqlite chunk-context cache should not evict")

    monkeypatch.setattr(
        "rag_core.search.providers.cache_sqlite.evict_sqlite_cache_rows",
        _fail_eviction,
    )
    cache = SqliteChunkContextCache(tmp_path / "chunk_ctx.sqlite")
    key = _key(chunk_index=3)
    try:
        asyncio.run(cache.put(key, "context"))
        assert asyncio.run(cache.get(key)) == "context"
    finally:
        cache.close()


def test_sqlite_chunk_context_cache_expires_old_rows_after_reopen(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "chunk_ctx.sqlite"
    old_key = _key(chunk_index=0)
    fresh_key = _key(chunk_index=1)
    cache = SqliteChunkContextCache(db_path)
    try:
        asyncio.run(cache.put_many({old_key: "old", fresh_key: "fresh"}))
    finally:
        cache.close()
    _set_chunk_context_cache_ts(db_path, old_key, 1)

    reopened = SqliteChunkContextCache(db_path, max_age_seconds=10)
    try:
        result = asyncio.run(reopened.get_many([old_key, fresh_key]))
    finally:
        reopened.close()

    assert old_key not in result
    assert result[fresh_key] == "fresh"


def test_sqlite_chunk_context_cache_max_entries_trims_oldest_write_timestamp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = iter([10.0, 20.0, 30.0])
    monkeypatch.setattr(
        "rag_core.search.providers.chunk_context_cache.time.time",
        lambda: next(timestamps),
    )
    db_path = tmp_path / "chunk_ctx.sqlite"
    key_a = _key(chunk_index=0)
    key_b = _key(chunk_index=1)
    key_c = _key(chunk_index=2)
    cache = SqliteChunkContextCache(db_path, max_entries=2)
    try:
        asyncio.run(cache.put(key_a, "a"))
        asyncio.run(cache.put(key_b, "b"))
        assert asyncio.run(cache.get(key_a)) == "a"
        asyncio.run(cache.put(key_c, "c"))
        result = asyncio.run(cache.get_many([key_a, key_b, key_c]))
    finally:
        cache.close()

    assert key_a not in result
    assert result == {key_b: "b", key_c: "c"}


def test_caching_contextualizer_short_circuits_on_hit() -> None:
    inner = _RecordingContextualizer()
    wrapped = CachingContextualizer(inner, InMemoryChunkContextCache())

    first = asyncio.run(wrapped.contextualize(_request(chunk_index=0)))
    second = asyncio.run(wrapped.contextualize(_request(chunk_index=0)))

    assert first == second == "context-0"
    assert inner.calls == [0]


def test_caching_contextualizer_calls_inner_for_each_unique_chunk() -> None:
    inner = _RecordingContextualizer()
    wrapped = CachingContextualizer(inner, InMemoryChunkContextCache())

    asyncio.run(wrapped.contextualize(_request(chunk_index=0)))
    asyncio.run(wrapped.contextualize(_request(chunk_index=1)))

    assert inner.calls == [0, 1]


def test_caching_contextualizer_reindex_pays_no_call_through_sqlite(tmp_path: Path) -> None:
    inner = _RecordingContextualizer()
    cache = SqliteChunkContextCache(tmp_path / "ctx.sqlite")
    try:
        wrapped = CachingContextualizer(inner, cache)
        asyncio.run(wrapped.contextualize(_request(chunk_index=0)))
        asyncio.run(wrapped.contextualize(_request(chunk_index=1)))
        # Fresh wrapper with a fresh inner. Same persistent cache should serve both.
        reindex_inner = _RecordingContextualizer()
        wrapped_again = CachingContextualizer(reindex_inner, cache)
        result_a = asyncio.run(wrapped_again.contextualize(_request(chunk_index=0)))
        result_b = asyncio.run(wrapped_again.contextualize(_request(chunk_index=1)))
    finally:
        cache.close()

    assert inner.calls == [0, 1]
    assert reindex_inner.calls == []
    assert (result_a, result_b) == ("context-0", "context-1")


def test_caching_contextualizer_id_proxies_inner() -> None:
    wrapped = CachingContextualizer(_RecordingContextualizer(), InMemoryChunkContextCache())
    assert wrapped.contextualizer_id == "test:recording"


def test_caching_contextualizer_keys_on_document_hash() -> None:
    inner = _RecordingContextualizer()
    wrapped = CachingContextualizer(inner, InMemoryChunkContextCache())

    asyncio.run(wrapped.contextualize(_request(document_markdown="doc body")))
    asyncio.run(wrapped.contextualize(_request(document_markdown="different body")))

    assert inner.calls == [0, 0]
    assert sha256_text("doc body") != sha256_text("different body")


def test_caching_contextualizer_keys_on_document_filename() -> None:
    inner = _RecordingContextualizer()
    wrapped = CachingContextualizer(inner, InMemoryChunkContextCache())

    asyncio.run(wrapped.contextualize(_request()))
    renamed = _request()
    renamed = ChunkContextRequest(
        document_markdown=renamed.document_markdown,
        document_filename="renamed.md",
        chunk_text=renamed.chunk_text,
        chunk_index=renamed.chunk_index,
        total_chunks=renamed.total_chunks,
    )
    asyncio.run(wrapped.contextualize(renamed))

    assert inner.calls == [0, 0]


def test_caching_contextualizer_keys_on_chunk_identity_after_rechunk() -> None:
    inner = _RecordingContextualizer()
    wrapped = CachingContextualizer(inner, InMemoryChunkContextCache())

    asyncio.run(wrapped.contextualize(_request(chunk_index=0, total_chunks=2, chunk_text="alpha")))
    asyncio.run(wrapped.contextualize(_request(chunk_index=0, total_chunks=3, chunk_text="beta")))

    assert inner.calls == [0, 0]


def test_caching_contextualizer_skips_cache_for_provider_error_empty_context() -> None:
    class _ErroringContextualizer:
        contextualizer_id = "test:erroring"

        def __init__(self) -> None:
            self.calls = 0
            self._last_failed = False

        async def contextualize(self, request: ChunkContextRequest) -> str:
            self.calls += 1
            self._last_failed = True
            return ""

        def should_cache_context(self, context: str) -> bool:
            return not (self._last_failed and context == "")

    inner = _ErroringContextualizer()
    wrapped = CachingContextualizer(inner, InMemoryChunkContextCache())

    first = asyncio.run(wrapped.contextualize(_request(chunk_index=0)))
    second = asyncio.run(wrapped.contextualize(_request(chunk_index=0)))

    assert first == ""
    assert second == ""
    assert inner.calls == 2
