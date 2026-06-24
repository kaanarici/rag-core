from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from rag_core.search.embedding_cache_diagnostics import (
    embed_query_with_cache_observation,
    embed_texts_with_cache_observation,
)
from rag_core.search.providers.cached_embedding import (
    DEFAULT_EMBEDDING_NORMALIZATION,
    EMBEDDING_OPERATION_QUERY,
    EMBEDDING_OPERATION_TEXTS,
    CachedEmbeddingProvider,
)
from rag_core.search.providers.embedding_cache import (
    InMemoryCache,
    NoCache,
    SqliteCache,
)
from rag_core.search.indexer_prepare import embed_dense_texts
from rag_core.search.providers.embedding_cache_models import (
    EmbedCacheKey,
    EmbeddingCache,
    EmbeddingDocumentScope,
    sha256_text,
)
from rag_core.search.providers.embedding_input_types import (
    EMBEDDING_INPUT_DOCUMENT,
    EMBEDDING_INPUT_QUERY,
)
from tests.support import FakeEmbeddingProvider


def _make_key(text: str = "alpha", *, model_id: str = "fake-embedding") -> EmbedCacheKey:
    return EmbedCacheKey(
        provider="fake",
        provider_config_fingerprint="",
        model=model_id,
        dimensions=4,
        input_type=EMBEDDING_INPUT_DOCUMENT,
        normalization=DEFAULT_EMBEDDING_NORMALIZATION,
        processing_fingerprint="pf:v1",
        content_sha256=sha256_text(text),
    )


def _set_embedding_cache_ts(
    db_path: Path,
    key: EmbedCacheKey,
    ts: int,
) -> None:
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            "UPDATE embedding_cache SET ts = ? WHERE key = ?",
            (ts, key.stringify()),
        )
        connection.commit()
    finally:
        connection.close()


class _AsyncBarrier:
    def __init__(self, parties: int) -> None:
        self._parties = parties
        self._waiting = 0
        self._released = asyncio.Event()

    async def wait(self) -> None:
        self._waiting += 1
        if self._waiting >= self._parties:
            self._released.set()
        await self._released.wait()


class _BarrierEmbeddingProvider(FakeEmbeddingProvider):
    def __init__(self, parties: int) -> None:
        super().__init__()
        self._barrier = _AsyncBarrier(parties)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.embed_texts_calls.append(list(texts))
        await self._barrier.wait()
        return [self._embed(text) for text in texts]

    async def embed_query(self, query: str) -> list[float]:
        self.embed_query_calls.append(query)
        await self._barrier.wait()
        return self._embed(query)


class _IdentityEmbeddingProvider(FakeEmbeddingProvider):
    """FakeEmbeddingProvider with overridable identity fields for cache-key tests."""

    def __init__(
        self,
        *,
        provider: str = "fake",
        model: str = "fake-embedding",
        cache_identity: str = "",
        vocabulary: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(vocabulary) if vocabulary else super().__init__()
        self._provider = provider
        self._model = model
        self._cache_identity = cache_identity

    @property
    def provider_name(self) -> str:
        return self._provider

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def cache_identity(self) -> str:
        return self._cache_identity


class _ReorderingEmbeddingProvider(FakeEmbeddingProvider):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.embed_texts_calls.append(list(texts))
        return [self._embed(text) for text in reversed(texts)]


class _BadQueryEmbeddingProvider(FakeEmbeddingProvider):
    async def embed_query(self, query: str) -> list[float]:
        self.embed_query_calls.append(query)
        return [0.1]


class _BatchOnlyCache:
    def __init__(self) -> None:
        self.store: dict[EmbedCacheKey, list[float]] = {}
        self.get_many_calls: list[list[EmbedCacheKey]] = []
        self.put_many_calls: list[dict[EmbedCacheKey, list[float]]] = []

    async def get(self, key: EmbedCacheKey) -> list[float] | None:
        raise AssertionError("cached embedding runtime should use batch get")

    async def put(self, key: EmbedCacheKey, vector: list[float]) -> None:
        raise AssertionError("cached embedding runtime should use batch put")

    async def get_many(
        self,
        keys: list[EmbedCacheKey],
    ) -> dict[EmbedCacheKey, list[float]]:
        self.get_many_calls.append(list(keys))
        return {
            key: list(self.store[key])
            for key in keys
            if key in self.store
        }

    async def put_many(self, items: dict[EmbedCacheKey, list[float]]) -> None:
        self.put_many_calls.append({key: list(vector) for key, vector in items.items()})
        for key, vector in items.items():
            self.store[key] = list(vector)


def test_embed_cache_key_stringify_is_stable() -> None:
    key = EmbedCacheKey(
        provider="openai",
        provider_config_fingerprint="endpoint:a",
        model="text-embedding-3-small",
        dimensions=1536,
        input_type=EMBEDDING_INPUT_DOCUMENT,
        normalization=DEFAULT_EMBEDDING_NORMALIZATION,
        processing_fingerprint='{"base_version":"v1","source_type":"file"}',
        content_sha256="abc",
    )

    assert json.loads(key.stringify()) == {
        "content_sha256": "abc",
        # Scope fields default to ``""`` for legacy un-scoped callers; the
        # right-to-forget delete path purges scoped rows and leaves un-scoped
        # rows in place.
        "collection": "",
        "dimensions": 1536,
        "document_id": "",
        "input_type": EMBEDDING_INPUT_DOCUMENT,
        "model": "text-embedding-3-small",
        "namespace": "",
        "normalization": DEFAULT_EMBEDDING_NORMALIZATION,
        "processing_fingerprint": '{"base_version":"v1","source_type":"file"}',
        "provider": "openai",
        "provider_config_fingerprint": "endpoint:a",
    }


def test_in_memory_cache_delete_by_document_scope() -> None:
    cache = InMemoryCache()
    scoped_a = EmbedCacheKey(
        provider="openai",
        provider_config_fingerprint="endpoint:a",
        model="text-embedding-3-small",
        dimensions=4,
        input_type=EMBEDDING_INPUT_DOCUMENT,
        normalization=DEFAULT_EMBEDDING_NORMALIZATION,
        processing_fingerprint="pf",
        content_sha256="abc",
        namespace="workspace-alpha",
        collection="restricted",
        document_id="doc-A",
    )
    scoped_b = EmbedCacheKey(
        provider="openai",
        provider_config_fingerprint="endpoint:a",
        model="text-embedding-3-small",
        dimensions=4,
        input_type=EMBEDDING_INPUT_DOCUMENT,
        normalization=DEFAULT_EMBEDDING_NORMALIZATION,
        processing_fingerprint="pf",
        content_sha256="abc",
        namespace="workspace-alpha",
        collection="restricted",
        document_id="doc-B",
    )
    asyncio.run(cache.put(scoped_a, [0.1, 0.2, 0.3, 0.4]))
    asyncio.run(cache.put(scoped_b, [0.5, 0.6, 0.7, 0.8]))

    removed = asyncio.run(
        cache.delete_by_document_scope(
            namespace="workspace-alpha",
            collection="restricted",
            document_id="doc-A",
        )
    )

    assert removed == 1
    assert asyncio.run(cache.get(scoped_a)) is None
    assert asyncio.run(cache.get(scoped_b)) is not None


def test_embed_dense_texts_scopes_cache_for_right_to_forget() -> None:
    """Document embeddings must land under their document scope so a
    right-to-forget delete actually purges them. Regression: the indexer built
    cache keys with empty scope, so ``delete_by_document_scope`` matched
    nothing and the embedding bytes survived a document delete.
    """
    cache = InMemoryCache()
    provider = CachedEmbeddingProvider(FakeEmbeddingProvider(), cache)
    scope = EmbeddingDocumentScope(
        namespace="workspace-alpha",
        collection="restricted",
        document_id="doc-A",
    )

    vectors, _ = asyncio.run(
        embed_dense_texts(
            provider,
            ["alpha chunk", "beta chunk"],
            processing_fingerprint="pf:v1",
            batch_size=8,
            scope=scope,
        )
    )
    assert len(vectors) == 2

    removed = asyncio.run(
        cache.delete_by_document_scope(
            namespace="workspace-alpha",
            collection="restricted",
            document_id="doc-A",
        )
    )
    assert removed == 2


def test_embed_dense_texts_without_scope_stays_unscoped() -> None:
    """No scope (query / ad-hoc embeds) keeps the empty-scope default, so a
    scoped delete for a real document does not purge them by accident.
    """
    cache = InMemoryCache()
    provider = CachedEmbeddingProvider(FakeEmbeddingProvider(), cache)
    asyncio.run(
        embed_dense_texts(
            provider,
            ["alpha chunk"],
            processing_fingerprint="pf:v1",
            batch_size=8,
        )
    )
    removed = asyncio.run(
        cache.delete_by_document_scope(
            namespace="workspace-alpha",
            collection="restricted",
            document_id="doc-A",
        )
    )
    assert removed == 0


def test_no_cache_always_misses() -> None:
    cache: EmbeddingCache = NoCache()
    assert asyncio.run(cache.get(_make_key())) is None
    asyncio.run(cache.put(_make_key(), [0.1, 0.2]))
    assert asyncio.run(cache.get(_make_key())) is None


def test_in_memory_cache_round_trips_a_vector() -> None:
    cache = InMemoryCache()
    key = _make_key()
    asyncio.run(cache.put(key, [0.1, 0.2, 0.3]))
    assert asyncio.run(cache.get(key)) == [pytest.approx(0.1), pytest.approx(0.2), pytest.approx(0.3)]


@pytest.mark.parametrize(
    "field, override",
    [
        ("model", "model-b"),
        ("provider", "voyage"),
        ("dimensions", 768),
        ("processing_fingerprint", "pf:v2"),
        ("content_sha256", sha256_text("other")),
    ],
    ids=["model", "provider", "dimensions", "processing-fingerprint", "content"],
)
def test_in_memory_cache_isolates_keys_per_identity_field(
    field: str, override: object
) -> None:
    cache = InMemoryCache()
    base = _make_key("shared")
    sibling = replace(base, **{field: override})  # type: ignore[arg-type]

    asyncio.run(cache.put(base, [1.0]))
    asyncio.run(cache.put(sibling, [2.0]))

    assert asyncio.run(cache.get(base)) == [pytest.approx(1.0)]
    assert asyncio.run(cache.get(sibling)) == [pytest.approx(2.0)]


def test_in_memory_cache_evicts_lru_at_max_entries() -> None:
    cache = InMemoryCache(max_entries=3)
    asyncio.run(cache.put(_make_key("a"), [1.0]))
    asyncio.run(cache.put(_make_key("b"), [2.0]))
    asyncio.run(cache.put(_make_key("c"), [3.0]))
    asyncio.run(cache.get(_make_key("a")))  # touch -> most recent
    asyncio.run(cache.put(_make_key("d"), [4.0]))  # evict "b"

    assert asyncio.run(cache.get(_make_key("a"))) == [pytest.approx(1.0)]
    assert asyncio.run(cache.get(_make_key("b"))) is None
    assert asyncio.run(cache.get(_make_key("c"))) == [pytest.approx(3.0)]
    assert asyncio.run(cache.get(_make_key("d"))) == [pytest.approx(4.0)]


@pytest.mark.parametrize("max_entries", [0, -1])
def test_in_memory_cache_rejects_non_positive_max_entries(max_entries: int) -> None:
    with pytest.raises(ValueError):
        InMemoryCache(max_entries=max_entries)


def test_sqlite_cache_round_trips_after_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "cache.sqlite"
    cache = SqliteCache(db_path)
    key = _make_key("persistent")
    asyncio.run(cache.put(key, [0.5, 0.25]))
    cache.close()

    reopened = SqliteCache(db_path)
    try:
        result = asyncio.run(reopened.get(key))
    finally:
        reopened.close()

    assert result == [pytest.approx(0.5), pytest.approx(0.25)]


def test_sqlite_cache_creates_private_db_file_and_parent(tmp_path: Path) -> None:
    db_path = tmp_path / "cache" / "embedding.sqlite"
    cache = SqliteCache(db_path)
    try:
        asyncio.run(cache.put(_make_key("private"), [0.5]))
    finally:
        cache.close()

    if os.name != "nt":
        assert db_path.stat().st_mode & 0o777 == 0o600
        assert (tmp_path / "cache").stat().st_mode & 0o777 == 0o700


def test_sqlite_cache_rejects_symlink_db_path(tmp_path: Path) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlink support unavailable")
    target = tmp_path / "target.sqlite"
    db_path = tmp_path / "cache.sqlite"
    db_path.symlink_to(target)
    cache = SqliteCache(db_path)

    with pytest.raises(ValueError, match="path must not be a symlink"):
        asyncio.run(cache.put(_make_key("private"), [0.5]))

    assert not target.exists()


def test_sqlite_cache_rejects_relative_path_from_symlinked_logical_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlink support unavailable")
    real_root = tmp_path / "real"
    real_root.mkdir()
    alias_root = tmp_path / "alias"
    alias_root.symlink_to(real_root, target_is_directory=True)
    monkeypatch.chdir(real_root)
    monkeypatch.setenv("PWD", str(alias_root))

    cache = SqliteCache(Path("cache.sqlite"))
    with pytest.raises(ValueError, match="symlinked PWD"):
        asyncio.run(cache.put(_make_key("private"), [0.5]))

    assert not (real_root / "cache.sqlite").exists()


@pytest.mark.parametrize("max_age_seconds", [0.0, -1.0, float("nan")])
def test_sqlite_cache_rejects_invalid_max_age_seconds(
    tmp_path: Path,
    max_age_seconds: float,
) -> None:
    with pytest.raises(ValueError, match="max_age_seconds"):
        SqliteCache(tmp_path / "cache.sqlite", max_age_seconds=max_age_seconds)


@pytest.mark.parametrize("max_entries", [0, -1])
def test_sqlite_cache_rejects_invalid_max_entries(
    tmp_path: Path,
    max_entries: int,
) -> None:
    with pytest.raises(ValueError, match="max_entries"):
        SqliteCache(tmp_path / "cache.sqlite", max_entries=max_entries)


def test_sqlite_cache_unbounded_get_put_skip_eviction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail_eviction(*args: object, **kwargs: object) -> int:
        raise AssertionError("unbounded sqlite cache should not evict")

    monkeypatch.setattr(
        "rag_core.search.providers.cache_sqlite.evict_sqlite_cache_rows",
        _fail_eviction,
    )
    cache = SqliteCache(tmp_path / "cache.sqlite")
    key = _make_key("unbounded")
    try:
        asyncio.run(cache.put(key, [0.25]))
        assert asyncio.run(cache.get(key)) == [pytest.approx(0.25)]
    finally:
        cache.close()


def test_sqlite_cache_expires_old_rows_after_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "cache.sqlite"
    old_key = _make_key("old")
    fresh_key = _make_key("fresh")
    cache = SqliteCache(db_path)
    try:
        asyncio.run(cache.put_many({old_key: [1.0], fresh_key: [2.0]}))
    finally:
        cache.close()
    _set_embedding_cache_ts(db_path, old_key, 1)

    reopened = SqliteCache(db_path, max_age_seconds=10)
    try:
        result = asyncio.run(reopened.get_many([old_key, fresh_key]))
    finally:
        reopened.close()

    assert old_key not in result
    assert result[fresh_key] == [pytest.approx(2.0)]


def test_sqlite_cache_max_entries_trims_oldest_write_timestamp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = iter([10.0, 20.0, 30.0])
    monkeypatch.setattr(
        "rag_core.search.providers.embedding_cache.time.time",
        lambda: next(timestamps),
    )
    db_path = tmp_path / "cache.sqlite"
    key_a = _make_key("a")
    key_b = _make_key("b")
    key_c = _make_key("c")
    cache = SqliteCache(db_path, max_entries=2)
    try:
        asyncio.run(cache.put(key_a, [1.0]))
        asyncio.run(cache.put(key_b, [2.0]))
        assert asyncio.run(cache.get(key_a)) == [pytest.approx(1.0)]
        asyncio.run(cache.put(key_c, [3.0]))
        result = asyncio.run(cache.get_many([key_a, key_b, key_c]))
    finally:
        cache.close()

    assert key_a not in result
    assert result[key_b] == [pytest.approx(2.0)]
    assert result[key_c] == [pytest.approx(3.0)]


def test_sqlite_cache_replaces_existing_vector(tmp_path: Path) -> None:
    cache = SqliteCache(tmp_path / "cache.sqlite")
    key = _make_key()
    try:
        asyncio.run(cache.put(key, [0.1]))
        asyncio.run(cache.put(key, [0.9]))
        assert asyncio.run(cache.get(key)) == [pytest.approx(0.9)]
    finally:
        cache.close()


def test_sqlite_cache_batch_get_put_round_trips(tmp_path: Path) -> None:
    cache = SqliteCache(tmp_path / "cache.sqlite")
    key_a = _make_key("a")
    key_b = _make_key("b")
    key_missing = _make_key("missing")
    try:
        asyncio.run(cache.put_many({key_a: [1.0], key_b: [2.0]}))
        result = asyncio.run(cache.get_many([key_a, key_b, key_missing]))
    finally:
        cache.close()

    assert result[key_a] == [pytest.approx(1.0)]
    assert result[key_b] == [pytest.approx(2.0)]
    assert key_missing not in result


def test_sqlite_cache_resets_incompatible_embedding_table(tmp_path: Path) -> None:
    db_path = tmp_path / "cache.sqlite"
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            "CREATE TABLE embedding_cache ("
            "key TEXT PRIMARY KEY, vector BLOB NOT NULL, "
            "model_id TEXT NOT NULL, ts INTEGER NOT NULL"
            ")"
        )
        connection.execute(
            "INSERT INTO embedding_cache (key, vector, model_id, ts) "
            "VALUES (?, ?, ?, ?)",
            ("old-key", b"old-vector", "old-model", 1),
        )
        connection.commit()
    finally:
        connection.close()

    cache = SqliteCache(db_path)
    key = _make_key("fresh")
    try:
        asyncio.run(cache.put(key, [0.25, 0.5]))
        assert asyncio.run(cache.get(key)) == [pytest.approx(0.25), pytest.approx(0.5)]
    finally:
        cache.close()

    connection = sqlite3.connect(db_path)
    try:
        schema = frozenset(
            (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
            for row in connection.execute("PRAGMA table_info(embedding_cache)").fetchall()
        )
        row_count = connection.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0]
    finally:
        connection.close()

    assert schema == SqliteCache._EXPECTED_SCHEMA
    assert row_count == 1


def test_sqlite_cache_treats_malformed_vector_blob_as_miss_and_deletes_row(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "cache.sqlite"
    cache = SqliteCache(db_path)
    key = _make_key("corrupt")
    try:
        asyncio.run(cache.put(key, [0.25, 0.5]))
    finally:
        cache.close()

    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            "UPDATE embedding_cache SET vector = ? WHERE key = ?",
            (b"\x01", key.stringify()),
        )
        connection.commit()
    finally:
        connection.close()

    reopened = SqliteCache(db_path)
    try:
        assert asyncio.run(reopened.get(key)) is None
    finally:
        reopened.close()

    connection = sqlite3.connect(db_path)
    try:
        row_count = connection.execute(
            "SELECT COUNT(*) FROM embedding_cache WHERE key = ?",
            (key.stringify(),),
        ).fetchone()[0]
    finally:
        connection.close()

    assert row_count == 0


def test_sqlite_cache_batch_get_deletes_malformed_vector_blob(tmp_path: Path) -> None:
    db_path = tmp_path / "cache.sqlite"
    cache = SqliteCache(db_path)
    corrupt_key = _make_key("corrupt")
    valid_key = _make_key("valid")
    try:
        asyncio.run(cache.put_many({corrupt_key: [0.25], valid_key: [0.5]}))
    finally:
        cache.close()

    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            "UPDATE embedding_cache SET vector = ? WHERE key = ?",
            (b"\x01", corrupt_key.stringify()),
        )
        connection.commit()
    finally:
        connection.close()

    reopened = SqliteCache(db_path)
    try:
        result = asyncio.run(reopened.get_many([corrupt_key, valid_key]))
    finally:
        reopened.close()

    assert corrupt_key not in result
    assert result[valid_key] == [pytest.approx(0.5)]

    connection = sqlite3.connect(db_path)
    try:
        row_count = connection.execute(
            "SELECT COUNT(*) FROM embedding_cache WHERE key = ?",
            (corrupt_key.stringify(),),
        ).fetchone()[0]
    finally:
        connection.close()

    assert row_count == 0


def test_cached_embedding_provider_calls_inner_only_once_per_text() -> None:
    inner = FakeEmbeddingProvider()
    cached = CachedEmbeddingProvider(inner, InMemoryCache())

    first = asyncio.run(cached.embed_texts(["fox", "context"]))
    second = asyncio.run(cached.embed_texts(["fox", "context"]))

    assert first == second
    assert inner.embed_texts_calls == [["fox", "context"]]


def test_cached_embedding_provider_only_embeds_misses_and_preserves_order() -> None:
    inner = FakeEmbeddingProvider()
    cached = CachedEmbeddingProvider(inner, InMemoryCache())

    asyncio.run(cached.embed_texts(["fox"]))
    result = asyncio.run(cached.embed_texts(["query", "fox", "context"]))

    assert inner.embed_texts_calls == [["fox"], ["query", "context"]]
    inner_direct = asyncio.run(inner.embed_texts(["query", "context"]))
    assert result[0] == inner_direct[0]
    assert result[2] == inner_direct[1]


def test_cached_embedding_provider_treats_bad_text_cache_hit_as_miss() -> None:
    inner = FakeEmbeddingProvider()
    cache = InMemoryCache()
    cached = CachedEmbeddingProvider(inner, cache)
    bad_key = cached._build_key(
        "fox",
        input_type=EMBEDDING_INPUT_DOCUMENT,
        processing_fingerprint="",
    )
    asyncio.run(cache.put(bad_key, [0.1]))

    result = asyncio.run(cached.embed_texts(["fox"]))

    assert result == asyncio.run(inner.embed_texts(["fox"]))
    assert inner.embed_texts_calls == [["fox"], ["fox"]]
    observation = cached.diagnostics.last_observation
    assert observation is not None
    assert (observation.cache_hits, observation.cache_misses, observation.cache_writes) == (0, 1, 1)


def test_cached_embedding_provider_dedupes_same_batch_misses_by_cache_key() -> None:
    inner = FakeEmbeddingProvider()
    cached = CachedEmbeddingProvider(inner, InMemoryCache())

    result = asyncio.run(cached.embed_texts(["fox", "fox", "context", "fox"]))

    assert inner.embed_texts_calls == [["fox", "context"]]
    assert result[0] == result[1] == result[3]
    observation = cached.diagnostics.last_observation
    assert observation is not None
    assert observation.input_count == 4
    assert (observation.cache_hits, observation.cache_misses, observation.cache_writes) == (0, 4, 2)


def test_cached_embedding_provider_batches_misses_before_cache_writes() -> None:
    inner = FakeEmbeddingProvider()
    cached = CachedEmbeddingProvider(inner, InMemoryCache())

    first = asyncio.run(cached.embed_texts(["fox", "query"]))
    second = asyncio.run(cached.embed_texts(["fox", "query"]))

    assert first == asyncio.run(FakeEmbeddingProvider().embed_texts(["fox", "query"]))
    assert second == first
    assert inner.embed_texts_calls == [["fox", "query"]]


def test_cached_embedding_provider_uses_cache_batch_methods_for_texts() -> None:
    inner = FakeEmbeddingProvider()
    cache = _BatchOnlyCache()
    cached = CachedEmbeddingProvider(inner, cache)

    first = asyncio.run(cached.embed_texts(["fox", "query"]))
    second = asyncio.run(cached.embed_texts(["query", "fox"]))

    assert first == list(reversed(second))
    assert inner.embed_texts_calls == [["fox", "query"]]
    assert len(cache.get_many_calls) == 2
    assert len(cache.put_many_calls) == 1
    assert len(cache.put_many_calls[0]) == 2


@pytest.mark.parametrize(
    "left_kwargs, right_kwargs",
    [
        ({"model": "model-a"}, {"model": "model-b"}),
        ({"provider": "openai"}, {"provider": "voyage"}),
        (
            {"provider": "openai", "vocabulary": ("original", "context", "fox", "query")},
            {"provider": "openai", "vocabulary": ("original", "context")},
        ),
        ({"cache_identity": "endpoint:a"}, {"cache_identity": "endpoint:b"}),
    ],
    ids=["model-id", "provider-name", "dimensions", "cache-identity"],
)
def test_cached_embedding_provider_keys_isolate_provider_identity(
    left_kwargs: dict, right_kwargs: dict
) -> None:
    cache = InMemoryCache()
    inner_left = _IdentityEmbeddingProvider(**left_kwargs)
    inner_right = _IdentityEmbeddingProvider(**right_kwargs)
    cached_left = CachedEmbeddingProvider(inner_left, cache)
    cached_right = CachedEmbeddingProvider(inner_right, cache)

    asyncio.run(cached_left.embed_texts(["shared"]))
    asyncio.run(cached_right.embed_texts(["shared"]))
    asyncio.run(cached_left.embed_texts(["shared"]))

    # Each identity computed once; cache hit on the third call to "left".
    assert inner_left.embed_texts_calls == [["shared"]]
    assert inner_right.embed_texts_calls == [["shared"]]


def test_cached_embedding_provider_keys_per_processing_fingerprint() -> None:
    cache = InMemoryCache()
    inner_v1 = _IdentityEmbeddingProvider(provider="openai")
    inner_v2 = _IdentityEmbeddingProvider(provider="openai")

    asyncio.run(
        CachedEmbeddingProvider(inner_v1, cache, processing_fingerprint="pf:v1").embed_texts(["shared"])
    )
    asyncio.run(
        CachedEmbeddingProvider(inner_v2, cache, processing_fingerprint="pf:v2").embed_texts(["shared"])
    )

    assert inner_v1.embed_texts_calls == [["shared"]]
    assert inner_v2.embed_texts_calls == [["shared"]]


@pytest.mark.parametrize(
    "cache_queries, expected_query_calls",
    [(False, ["fox", "fox"]), (True, ["fox"])],
    ids=["default-bypass", "opt-in-cache"],
)
def test_cached_embedding_provider_query_caching_is_opt_in(
    cache_queries: bool, expected_query_calls: list[str]
) -> None:
    inner = FakeEmbeddingProvider()
    cached = CachedEmbeddingProvider(inner, InMemoryCache(), cache_queries=cache_queries)

    asyncio.run(cached.embed_query("fox"))
    asyncio.run(cached.embed_query("fox"))

    assert inner.embed_query_calls == expected_query_calls


def test_cached_embedding_provider_treats_bad_query_cache_hit_as_miss() -> None:
    inner = FakeEmbeddingProvider()
    cache = InMemoryCache()
    cached = CachedEmbeddingProvider(inner, cache, cache_queries=True)
    bad_key = cached._build_key(
        "fox",
        input_type=EMBEDDING_INPUT_QUERY,
        processing_fingerprint="",
    )
    asyncio.run(cache.put(bad_key, [0.1]))

    result = asyncio.run(cached.embed_query("fox"))

    assert result == asyncio.run(inner.embed_query("fox"))
    assert inner.embed_query_calls == ["fox", "fox"]
    observation = cached.diagnostics.last_observation
    assert observation is not None
    assert (observation.cache_hits, observation.cache_misses, observation.cache_writes) == (0, 1, 1)


def test_cached_embedding_provider_rejects_invalid_fresh_query_vector_on_bypass() -> None:
    inner = _BadQueryEmbeddingProvider()
    cached = CachedEmbeddingProvider(inner, InMemoryCache(), cache_queries=False)

    with pytest.raises(ValueError, match="embedding dimension mismatch"):
        asyncio.run(cached.embed_query("fox"))


def test_cached_embedding_provider_rejects_invalid_fresh_query_vector_before_cache_write() -> (
    None
):
    inner = _BadQueryEmbeddingProvider()
    cache = InMemoryCache()
    cached = CachedEmbeddingProvider(inner, cache, cache_queries=True)
    key = cached._build_key(
        "fox",
        input_type=EMBEDDING_INPUT_QUERY,
        processing_fingerprint="",
    )

    with pytest.raises(ValueError, match="embedding dimension mismatch"):
        asyncio.run(cached.embed_query("fox"))

    assert asyncio.run(cache.get(key)) is None


def test_cached_embedding_provider_keeps_query_and_document_cache_keys_separate() -> None:
    inner = FakeEmbeddingProvider()
    cached = CachedEmbeddingProvider(inner, InMemoryCache(), cache_queries=True)

    asyncio.run(cached.embed_texts(["fox"]))
    asyncio.run(cached.embed_query("fox"))

    assert inner.embed_texts_calls == [["fox"]]
    assert inner.embed_query_calls == ["fox"]


def test_cached_embedding_provider_returns_empty_for_empty_input() -> None:
    inner = FakeEmbeddingProvider()
    cached = CachedEmbeddingProvider(inner, InMemoryCache())

    assert asyncio.run(cached.embed_texts([])) == []
    assert inner.embed_texts_calls == []


def test_cached_embedding_provider_dimensions_proxy_inner() -> None:
    inner = FakeEmbeddingProvider()
    cached = CachedEmbeddingProvider(inner, InMemoryCache())

    assert cached.dimensions == inner.dimensions
    assert cached.model_name == inner.model_name


def test_cached_embedding_provider_reports_text_diagnostics() -> None:
    inner = FakeEmbeddingProvider()
    cached = CachedEmbeddingProvider(inner, InMemoryCache())

    asyncio.run(cached.embed_texts(["fox", "query"]))
    asyncio.run(cached.embed_texts(["fox", "query", "context"]))

    diagnostics = cached.diagnostics
    assert (diagnostics.text_requests, diagnostics.query_requests) == (2, 0)
    assert (diagnostics.cache_hits, diagnostics.cache_misses, diagnostics.cache_writes) == (2, 3, 3)
    assert diagnostics.query_bypasses == 0
    observation = diagnostics.last_observation
    assert observation is not None
    assert observation.operation == EMBEDDING_OPERATION_TEXTS
    assert observation.input_count == 3
    assert (observation.cache_hits, observation.cache_misses, observation.cache_writes) == (2, 1, 1)
    assert observation.cache_bypassed is False


@pytest.mark.parametrize(
    "cache_queries, expected_bypasses, expected_hits, expected_misses, expected_writes, expected_bypassed",
    [
        (False, 2, 0, 0, 0, True),
        (True, 0, 1, 1, 1, False),
    ],
    ids=["bypass-by-default", "cache-when-enabled"],
)
def test_cached_embedding_provider_reports_query_diagnostics(
    cache_queries: bool,
    expected_bypasses: int,
    expected_hits: int,
    expected_misses: int,
    expected_writes: int,
    expected_bypassed: bool,
) -> None:
    inner = FakeEmbeddingProvider()
    cached = CachedEmbeddingProvider(inner, InMemoryCache(), cache_queries=cache_queries)

    asyncio.run(cached.embed_query("fox"))
    asyncio.run(cached.embed_query("fox"))

    diagnostics = cached.diagnostics
    assert (diagnostics.text_requests, diagnostics.query_requests) == (0, 2)
    assert diagnostics.query_bypasses == expected_bypasses
    assert (diagnostics.cache_hits, diagnostics.cache_misses, diagnostics.cache_writes) == (
        expected_hits,
        expected_misses,
        expected_writes,
    )
    observation = diagnostics.last_observation
    assert observation is not None
    assert observation.operation == EMBEDDING_OPERATION_QUERY
    assert observation.cache_bypassed is expected_bypassed


def test_cache_observation_helpers_are_call_scoped_under_concurrency() -> None:
    async def _run() -> None:
        text_provider = CachedEmbeddingProvider(
            _BarrierEmbeddingProvider(parties=2), InMemoryCache()
        )
        text_results = await asyncio.gather(
            embed_texts_with_cache_observation(text_provider, ["fox"]),
            embed_texts_with_cache_observation(text_provider, ["query"]),
        )
        text_counters = [counters for _, counters in text_results]
        assert [c.hits for c in text_counters] == [0, 0]
        assert [c.misses for c in text_counters] == [1, 1]
        assert [c.writes for c in text_counters] == [1, 1]

        query_provider = CachedEmbeddingProvider(
            _BarrierEmbeddingProvider(parties=2), InMemoryCache()
        )
        query_results = await asyncio.gather(
            embed_query_with_cache_observation(query_provider, "fox"),
            embed_query_with_cache_observation(query_provider, "query"),
        )
        query_counters = [counters for _, counters in query_results]
        assert [c.bypasses for c in query_counters] == [1, 1]
        assert [c.hits for c in query_counters] == [0, 0]
        assert [c.misses for c in query_counters] == [0, 0]
        assert [c.writes for c in query_counters] == [0, 0]

    asyncio.run(_run())


def test_cached_embedding_provider_propagates_inner_failures() -> None:
    class _FailingProvider(FakeEmbeddingProvider):
        async def embed_texts(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("boom")

        async def embed_query(self, query: str) -> list[float]:
            raise RuntimeError("query boom")

    cached = CachedEmbeddingProvider(_FailingProvider(), InMemoryCache())
    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(cached.embed_texts(["fox"]))
    assert cached.diagnostics.text_requests == 0

    with pytest.raises(RuntimeError, match="query boom"):
        asyncio.run(cached.embed_query("fox"))
    assert cached.diagnostics.query_requests == 0


def test_cached_embedding_provider_provider_name_falls_back_to_class_name() -> None:
    class _Plain:
        @property
        def dimensions(self) -> int:
            return 1

        @property
        def model_name(self) -> str:
            return "x"

        async def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] for _ in texts]

        async def embed_query(self, query: str) -> list[float]:
            return [0.0]

    cached = CachedEmbeddingProvider(_Plain(), InMemoryCache())
    assert cached.provider_name == "_Plain"


def test_cached_embedding_provider_uses_explicit_provider_name() -> None:
    inner = FakeEmbeddingProvider()
    inner.provider_name = "voyage"  # type: ignore[attr-defined]
    cached = CachedEmbeddingProvider(inner, InMemoryCache())
    assert cached.provider_name == "voyage"


def test_sha256_text_is_stable_and_collision_free_for_simple_inputs() -> None:
    assert sha256_text("a") == sha256_text("a")
    assert sha256_text("a") != sha256_text("b")
