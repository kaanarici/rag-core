"""Concurrency / WAL / per-tier path tests for the cache providers.

Proof labels (per tests/README.md):

- ``test_sqlite_cache_one_hundred_concurrent_tasks_round_trip``: plumbing.
  Exercises the per-task connection pool against the real ``sqlite3``
  module on the real cache file. Does not call any embedding provider.
- ``test_sqlite_cache_enables_wal_and_busy_timeout``: plumbing. Inspects
  the live PRAGMA on the open connection.
- ``test_sqlite_chunk_context_cache_one_hundred_concurrent_tasks_round_trip``:
  plumbing. Same shape against ``SqliteChunkContextCache``.
- ``test_build_cache_path_refuses_shared_path_across_scope``: contract.
  The path helper is the gate for per-tier isolation.
- ``test_collection_policy_cache_disabled_swaps_in_no_cache``: contract. The
  restricted-tier deploy switch must replace the configured cache.
- ``test_open_sqlite_cache_hardens_wal_sidecar_files`` / ``..._shm``:
  plumbing. Exercises POSIX perms on the materialized WAL/SHM files.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import threading
from pathlib import Path

import pytest

from rag_core.search.providers.cache_sqlite_connection import (
    SqliteCacheConnectionPool,
    bind_running_task_for_pool,
    build_cache_path,
    harden_sqlite_sidecar_files,
    open_sqlite_cache,
)
from rag_core.search.providers.chunk_context_cache import (
    ChunkContextKey,
    SqliteChunkContextCache,
)
from rag_core.search.providers.embedding_cache import SqliteCache
from rag_core.search.providers.embedding_cache_models import (
    EmbedCacheKey,
    sha256_text,
)
from rag_core.search.providers.embedding_input_types import EMBEDDING_INPUT_DOCUMENT


def _embed_key(text: str) -> EmbedCacheKey:
    return EmbedCacheKey(
        provider="fake",
        provider_config_fingerprint="",
        model="fake-embedding",
        dimensions=4,
        input_type=EMBEDDING_INPUT_DOCUMENT,
        normalization="none",
        processing_fingerprint="pf:v1",
        content_sha256=sha256_text(text),
    )


def _context_key(index: int) -> ChunkContextKey:
    return ChunkContextKey(
        contextualizer_id="ctx",
        document_sha256=f"doc{index}",
        document_filename_sha256="filename",
        chunk_text_sha256=f"chunk{index}",
        chunk_index=index,
        total_chunks=100,
        namespace="",
        collection="",
        document_id="",
    )


# ----------------------------------------------------------------------
# WAL / pragma / private-file plumbing


def test_open_sqlite_cache_enables_wal_and_busy_timeout(tmp_path: Path) -> None:
    db_path = tmp_path / "cache.sqlite"
    connection = open_sqlite_cache(db_path)
    try:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        busy_timeout = connection.execute("PRAGMA busy_timeout").fetchone()[0]
    finally:
        connection.close()
    assert str(journal_mode).lower() == "wal"
    assert int(busy_timeout) >= 5000


def test_open_sqlite_cache_sets_check_same_thread_false(tmp_path: Path) -> None:
    db_path = tmp_path / "cache.sqlite"
    connection = open_sqlite_cache(db_path)
    try:
        captured: list[Exception | None] = [None]

        def _ping() -> None:
            try:
                connection.execute("SELECT 1").fetchone()
            except Exception as exc:  # noqa: BLE001 - capturing for assertion
                captured[0] = exc

        thread = threading.Thread(target=_ping)
        thread.start()
        thread.join()
    finally:
        connection.close()
    assert captured[0] is None


def test_open_sqlite_cache_hardens_db_and_wal_sidecar_files(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("POSIX-only perm check")
    db_path = tmp_path / "cache.sqlite"
    connection = open_sqlite_cache(db_path)
    try:
        # Force a write so WAL/SHM materialize.
        connection.execute("CREATE TABLE t (x INTEGER)")
        connection.execute("INSERT INTO t VALUES (1)")
        connection.commit()
        harden_sqlite_sidecar_files(db_path)
    finally:
        connection.close()
    assert db_path.stat().st_mode & 0o777 == 0o600
    for suffix in ("-wal", "-shm"):
        sidecar = db_path.with_name(db_path.name + suffix)
        if sidecar.exists():
            assert sidecar.stat().st_mode & 0o777 == 0o600


def test_open_sqlite_cache_rejects_symlink_db_path(tmp_path: Path) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlink support unavailable")
    target = tmp_path / "target.sqlite"
    db_path = tmp_path / "cache.sqlite"
    db_path.symlink_to(target)
    with pytest.raises(ValueError, match="path must not be a symlink"):
        open_sqlite_cache(db_path)


def test_harden_sqlite_sidecar_files_rejects_hardlinked_wal(tmp_path: Path) -> None:
    if os.name == "nt" or not hasattr(os, "link"):
        pytest.skip("POSIX-only hardlink check")
    db_path = tmp_path / "cache.sqlite"
    db_path.write_bytes(b"")
    wal_path = db_path.with_name(db_path.name + "-wal")
    wal_path.write_bytes(b"")
    extra = tmp_path / "extra"
    os.link(wal_path, extra)
    with pytest.raises(ValueError, match="must not be hard-linked"):
        harden_sqlite_sidecar_files(db_path)


# ----------------------------------------------------------------------
# Per-task connection pool. 100-task concurrent ingest


def test_sqlite_cache_one_hundred_concurrent_tasks_round_trip(tmp_path: Path) -> None:
    db_path = tmp_path / "cache.sqlite"
    cache = SqliteCache(db_path)

    async def _writer(index: int) -> None:
        key = _embed_key(f"chunk-{index}")
        await cache.put(key, [float(index), float(index) + 0.5, 0.0, 0.0])
        result = await cache.get(key)
        assert result == [
            pytest.approx(float(index)),
            pytest.approx(float(index) + 0.5),
            pytest.approx(0.0),
            pytest.approx(0.0),
        ]

    async def _run() -> None:
        await asyncio.gather(*(_writer(i) for i in range(100)))

    try:
        # ProgrammingError on cross-task connection sharing would surface
        # here. WAL durability is verified by the post-close reopen check.
        asyncio.run(_run())
    finally:
        cache.close()

    # Verify the file is not corrupt and every row landed.
    reopened = sqlite3.connect(db_path)
    try:
        row_count = reopened.execute(
            "SELECT COUNT(*) FROM embedding_cache"
        ).fetchone()[0]
        integrity = reopened.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        reopened.close()
    assert int(row_count) == 100
    assert str(integrity).lower() == "ok"


def test_sqlite_chunk_context_cache_one_hundred_concurrent_tasks_round_trip(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "context.sqlite"
    cache = SqliteChunkContextCache(db_path)

    async def _writer(index: int) -> None:
        key = _context_key(index)
        await cache.put(key, f"context-{index}")
        assert await cache.get(key) == f"context-{index}"

    async def _run() -> None:
        await asyncio.gather(*(_writer(i) for i in range(100)))

    try:
        asyncio.run(_run())
    finally:
        cache.close()

    reopened = sqlite3.connect(db_path)
    try:
        row_count = reopened.execute(
            "SELECT COUNT(*) FROM chunk_context_cache"
        ).fetchone()[0]
        integrity = reopened.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        reopened.close()
    assert int(row_count) == 100
    assert str(integrity).lower() == "ok"


def test_sqlite_cache_concurrent_get_many_does_not_raise(tmp_path: Path) -> None:
    db_path = tmp_path / "cache.sqlite"
    cache = SqliteCache(db_path)
    keys = [_embed_key(f"k{i}") for i in range(40)]

    async def _seed() -> None:
        await cache.put_many({key: [float(i), 0.0, 0.0, 0.0] for i, key in enumerate(keys)})

    async def _reader(_: int) -> None:
        result = await cache.get_many(keys)
        assert len(result) == len(keys)

    async def _run() -> None:
        await _seed()
        await asyncio.gather(*(_reader(i) for i in range(64)))

    try:
        asyncio.run(_run())
    finally:
        cache.close()


def test_sqlite_cache_pool_uses_distinct_connection_per_task(tmp_path: Path) -> None:
    pool = SqliteCacheConnectionPool(tmp_path / "pool.sqlite")
    seen: list[int] = []
    seen_lock = asyncio.Lock()

    async def _record() -> None:
        # Production path: cache providers call ``bind_running_task_for_pool``
        # in the awaiting coroutine so ``asyncio.to_thread`` copies the task
        # contextvar into the worker, letting the pool key by task id.
        bind_running_task_for_pool()
        connection_id = await asyncio.to_thread(_leased_connection_id, pool)
        async with seen_lock:
            seen.append(connection_id)

    async def _run() -> None:
        await asyncio.gather(*(_record() for _ in range(20)))

    try:
        asyncio.run(_run())
    finally:
        pool.close()

    # 20 concurrent asyncio tasks should each receive a distinct connection
    # object (per-task pool key). The set guarantees no sharing.
    assert len(set(seen)) == 20


def _leased_connection_id(pool: SqliteCacheConnectionPool) -> int:
    with pool.connection() as connection:
        return id(connection)


def test_sqlite_cache_worker_keeps_connection_until_thread_exits(
    tmp_path: Path,
) -> None:
    cache = SqliteCache(tmp_path / "cache.sqlite")
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    worker_errors: list[Exception] = []

    def blocked_write(connection: sqlite3.Connection) -> None:
        try:
            entered.set()
            assert release.wait(timeout=5)
            connection.execute("CREATE TABLE worker_probe (value INTEGER)")
            connection.execute("INSERT INTO worker_probe VALUES (1)")
            connection.commit()
        except Exception as exc:  # noqa: BLE001 - captured for assertion.
            worker_errors.append(exc)
        finally:
            finished.set()

    async def _run() -> None:
        task = asyncio.create_task(cache._run(blocked_write))
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        assert await asyncio.to_thread(finished.wait, 5)

    try:
        asyncio.run(_run())
    finally:
        cache.close()

    assert worker_errors == []


# ----------------------------------------------------------------------
# Per-tier path scoping


def test_build_cache_path_segments_by_fingerprint_embedder_and_corpus(
    tmp_path: Path,
) -> None:
    public = build_cache_path(
        base_dir=tmp_path,
        processing_fingerprint="pf:v1",
        embedder_identity="openai/text-embedding-3-large",
        collection="public",
    )
    restricted = build_cache_path(
        base_dir=tmp_path,
        processing_fingerprint="pf:v1",
        embedder_identity="openai/text-embedding-3-large",
        collection="restricted",
    )
    assert public != restricted
    assert public.parent.name == "public"
    assert restricted.parent.name == "restricted"
    assert public.name == "cache.db"


def test_build_cache_path_refuses_empty_components(tmp_path: Path) -> None:
    for kwargs in (
        {"processing_fingerprint": ""},
        {"embedder_identity": "  "},
        {"collection": ""},
    ):
        defaults = dict(
            base_dir=tmp_path,
            processing_fingerprint="pf:v1",
            embedder_identity="emb",
            collection="public",
        )
        defaults.update(kwargs)
        with pytest.raises(ValueError, match="non-empty"):
            build_cache_path(**defaults)  # type: ignore[arg-type]


def test_build_cache_path_sanitizes_unsafe_segments(tmp_path: Path) -> None:
    path = build_cache_path(
        base_dir=tmp_path,
        processing_fingerprint="pf v1/extra",
        embedder_identity="emb/with:colon",
        collection="licensed",
    )
    # Path segments must not contain path separators or colons that would
    # let a malicious collection escape the base directory.
    segments = path.relative_to(tmp_path).parts[:-1]
    for segment in segments:
        for ch in segment:
            assert ch.isalnum() or ch in ("-", "_", ".")


# ----------------------------------------------------------------------
# Restricted-tier cache_disabled deploy switch


def test_collection_policy_cache_disabled_default_false() -> None:
    from rag_core.search.policy import CollectionPolicy

    assert CollectionPolicy().cache_disabled is False


def test_collection_policy_cache_disabled_swaps_to_no_cache(tmp_path: Path) -> None:
    """When ``cache_disabled=True`` the assembler must hand the embedding
    provider a NoCache regardless of the IngestConfig.

    This is the restricted-tier deploy contract. Sensitive paraphrases
    cannot land on disk in a cache file.
    """

    from rag_core import Engine
    from rag_core.config import IngestConfig
    from rag_core.search.policy import CollectionPolicy
    from rag_core.search.providers.cache_sqlite import SQLITE_CACHE_PROVIDER
    from rag_core.search.providers.cached_embedding import CachedEmbeddingProvider
    from rag_core.search.providers.embedding_cache import NoCache
    from tests.support import (
        FakeEmbeddingProvider,
        FakeSparseEmbedder,
        RecordingVectorStore,
        make_test_config,
    )

    base = make_test_config(
        qdrant_collection="rag_core_restricted_cache_disabled",
        embedding_dimensions=4,
    )
    config = type(base)(
        qdrant=base.qdrant,
        embedding=base.embedding,
        reranker=base.reranker,
        chunking=base.chunking,
        ingest=IngestConfig(
            processing_version=base.ingest.processing_version,
            source_type=base.ingest.source_type,
            enable_lexical_search=False,
            manifest_directory=base.ingest.manifest_directory,
            embedding_cache_provider=SQLITE_CACHE_PROVIDER,
            embedding_cache_path=tmp_path / "would_have_been_sqlite.db",
        ),
        policy=base.policy,
        collection_policy=CollectionPolicy(
            bound_namespace="signal-workspace-1",
            allowed_collections=frozenset({"restricted"}),
            allow_rerank=False,
            allow_lexical_sidecar=False,
            cache_disabled=True,
        ),
    )
    core = Engine(
        config,
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=RecordingVectorStore(),
    )
    try:
        assert isinstance(core._embedding, CachedEmbeddingProvider)
        assert isinstance(core._embedding._cache, NoCache)
        # The configured sqlite path must NOT have been touched.
        assert not (tmp_path / "would_have_been_sqlite.db").exists()
    finally:
        asyncio.run(core.close())
