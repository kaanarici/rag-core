from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from rag_core.search.providers.chunk_context_cache import (
    ChunkContextCache,
    ChunkContextKey,
    InMemoryChunkContextCache,
    NoChunkContextCache,
    SqliteChunkContextCache,
    create_chunk_context_cache,
)
from rag_core.search.providers.cache_provider_names import (
    CACHE_PROVIDER_ORDER,
    IN_MEMORY_CACHE_PROVIDER,
    NO_CACHE_PROVIDER,
    SQLITE_CACHE_PROVIDER,
)
from rag_core.search.providers.embedding_cache import (
    DEFAULT_EMBEDDING_CACHE_PROVIDER,
    InMemoryCache,
    NoCache,
    SqliteCache,
    create_embedding_cache,
)
from rag_core.search.providers.embedding_cache_models import (
    EmbedCacheKey,
    EmbeddingCache,
)
from rag_core.search.providers.registry import (
    CHUNK_CONTEXT_CACHES,
    EMBEDDING_CACHES,
    ProviderRegistry,
)


def _embed_key() -> EmbedCacheKey:
    return EmbedCacheKey(
        provider="fake",
        provider_config_fingerprint="",
        model="fake-embedding",
        dimensions=4,
        input_type="document",
        normalization="text_sha256_utf8",
        processing_fingerprint="pf",
        content_sha256="sha",
    )


def test_cache_provider_defaults_are_owned_by_cache_factory_module() -> None:
    assert (
        DEFAULT_EMBEDDING_CACHE_PROVIDER
        == NO_CACHE_PROVIDER
        == NoCache.provider_name
    )
    assert InMemoryCache.provider_name == IN_MEMORY_CACHE_PROVIDER
    assert SqliteCache.provider_name == SQLITE_CACHE_PROVIDER
    assert NoChunkContextCache.provider_name == NO_CACHE_PROVIDER
    assert InMemoryChunkContextCache.provider_name == IN_MEMORY_CACHE_PROVIDER
    assert SqliteChunkContextCache.provider_name == SQLITE_CACHE_PROVIDER
    assert CACHE_PROVIDER_ORDER == (
        NO_CACHE_PROVIDER,
        IN_MEMORY_CACHE_PROVIDER,
        SQLITE_CACHE_PROVIDER,
    )


def test_chunk_context_cache_factory_is_owned_by_chunk_context_module() -> None:
    registry_source = Path("src/rag_core/search/providers/registry.py").read_text(
        encoding="utf-8"
    )
    embedding_cache_source = Path(
        "src/rag_core/search/providers/embedding_cache.py"
    ).read_text(encoding="utf-8")

    assert "rag_core.search.providers.chunk_context_cache" in registry_source
    assert (
        "CHUNK_CONTEXT_CACHES: ProviderRegistry"
        in registry_source
        and "rag_core.search.providers.embedding_cache" not in registry_source.split(
            "CHUNK_CONTEXT_CACHES: ProviderRegistry",
            1,
        )[1]
    )
    assert "CHUNK_CONTEXT_CACHES.register" not in embedding_cache_source
    assert "create_chunk_context_cache" not in embedding_cache_source


def test_provider_category_diagnostics_use_shared_cache_provider_names() -> None:
    diagnostics_source = Path(
        "src/rag_core/search/providers/provider_category_diagnostics.py"
    ).read_text(encoding="utf-8")

    assert "from .cache_provider_names import CACHE_PROVIDER_ORDER" in diagnostics_source
    assert "NO_CACHE_PROVIDER" in diagnostics_source
    assert "from .embedding_cache import" not in diagnostics_source


@pytest.mark.parametrize(
    "registry, expected_names",
    [
        (EMBEDDING_CACHES, set(CACHE_PROVIDER_ORDER)),
        (CHUNK_CONTEXT_CACHES, set(CACHE_PROVIDER_ORDER)),
    ],
    ids=["embedding-caches", "chunk-context-caches"],
)
def test_cache_registries_expose_builtin_names(
    registry: ProviderRegistry, expected_names: set[str]
) -> None:
    assert isinstance(registry, ProviderRegistry)
    assert expected_names <= set(registry.names())


@pytest.mark.parametrize(
    "name, sqlite_path, expected",
    [
        (NO_CACHE_PROVIDER, None, NoCache),
        (IN_MEMORY_CACHE_PROVIDER, None, InMemoryCache),
        (SQLITE_CACHE_PROVIDER, "cache.sqlite", SqliteCache),
        (None, None, NoCache),
    ],
    ids=["none", "in-memory", "sqlite", "implicit-none"],
)
def test_create_embedding_cache_returns_concrete_types(
    tmp_path: Path, name: str | None, sqlite_path: str | None, expected: type
) -> None:
    kwargs = {"path": tmp_path / sqlite_path} if sqlite_path else {}
    assert isinstance(create_embedding_cache(name, **kwargs), expected)


@pytest.mark.parametrize(
    "name, sqlite_path, expected",
    [
        (NO_CACHE_PROVIDER, None, NoChunkContextCache),
        (IN_MEMORY_CACHE_PROVIDER, None, InMemoryChunkContextCache),
        (SQLITE_CACHE_PROVIDER, "context.sqlite", SqliteChunkContextCache),
        (None, None, NoChunkContextCache),
    ],
    ids=["none", "in-memory", "sqlite", "implicit-none"],
)
def test_create_chunk_context_cache_returns_concrete_types(
    tmp_path: Path, name: str | None, sqlite_path: str | None, expected: type
) -> None:
    kwargs = {"path": tmp_path / sqlite_path} if sqlite_path else {}
    assert isinstance(create_chunk_context_cache(name, **kwargs), expected)


@pytest.mark.parametrize(
    "factory, message",
    [
        (create_embedding_cache, "Unknown embedding_cache provider"),
        (create_chunk_context_cache, "Unknown chunk_context_cache provider"),
    ],
    ids=["embedding-cache", "chunk-context-cache"],
)
def test_create_cache_unknown_name_raises(factory, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory("not-a-real-cache")


def test_user_registered_embedding_cache_works_end_to_end() -> None:
    class _StubCache:
        async def get(self, key: EmbedCacheKey) -> list[float] | None:
            return None

        async def put(self, key: EmbedCacheKey, vector: list[float]) -> None:
            return None

    name = "stub-test-embedding-cache"
    EMBEDDING_CACHES.register(name, lambda **kwargs: _StubCache())
    try:
        cache: EmbeddingCache = create_embedding_cache(name)
        assert isinstance(cache, _StubCache)
    finally:
        EMBEDDING_CACHES.unregister(name)


def test_user_registered_chunk_context_cache_works_end_to_end() -> None:
    class _StubContextCache:
        async def get(self, key: ChunkContextKey) -> str | None:
            return None

        async def get_many(self, keys: list[ChunkContextKey]) -> dict[ChunkContextKey, str]:
            return {}

        async def put(self, key: ChunkContextKey, context: str) -> None:
            return None

        async def put_many(self, items: dict[ChunkContextKey, str]) -> None:
            return None

    name = "stub-test-context-cache"
    CHUNK_CONTEXT_CACHES.register(name, lambda **kwargs: _StubContextCache())
    try:
        cache: ChunkContextCache = create_chunk_context_cache(name)
        assert isinstance(cache, _StubContextCache)
    finally:
        CHUNK_CONTEXT_CACHES.unregister(name)


def test_rag_core_resolves_embedding_cache_provider_from_config() -> None:
    """RAGCoreConfig.ingest.embedding_cache_provider wires the cache via factory."""
    from rag_core import RAGCore
    from rag_core.config import IngestConfig
    from rag_core.search.providers.cached_embedding import CachedEmbeddingProvider
    from tests.support import (
        FakeEmbeddingProvider,
        FakeSparseEmbedder,
        RecordingVectorStore,
        make_test_config,
    )

    base = make_test_config(
        qdrant_collection="rag_core_cache_provider_test",
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
            enable_lexical_search=base.ingest.enable_lexical_search,
            manifest_directory=base.ingest.manifest_directory,
            embedding_cache_provider=IN_MEMORY_CACHE_PROVIDER,
        ),
        policy=base.policy,
    )
    core = RAGCore(
        config,
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=RecordingVectorStore(),
    )
    try:
        # Private surface — RAGCore has no public introspection for cache wiring yet.
        # Worth promoting to public diagnostics so the config wiring can be tested observably.
        assert isinstance(core._embedding, CachedEmbeddingProvider)
        assert isinstance(core._embedding._cache, InMemoryCache)
        assert (
            core._embedding._processing_fingerprint
            == '{"base_version":"rag_core_processing_v1","source_type":"file"}'
        )
    finally:
        asyncio.run(core.close())


def test_rag_core_resolves_sqlite_embedding_cache_path_from_config(
    tmp_path: Path,
) -> None:
    from rag_core import RAGCore
    from rag_core.config import IngestConfig
    from rag_core.search.providers.cached_embedding import CachedEmbeddingProvider
    from tests.support import (
        FakeEmbeddingProvider,
        FakeSparseEmbedder,
        RecordingVectorStore,
        make_test_config,
    )

    base = make_test_config(
        qdrant_collection="rag_core_cache_provider_test",
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
            enable_lexical_search=base.ingest.enable_lexical_search,
            manifest_directory=base.ingest.manifest_directory,
            embedding_cache_provider=SQLITE_CACHE_PROVIDER,
            embedding_cache_path=tmp_path / "embeddings.sqlite",
        ),
        policy=base.policy,
    )
    core = RAGCore(
        config,
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=RecordingVectorStore(),
    )
    try:
        assert isinstance(core._embedding, CachedEmbeddingProvider)
        assert isinstance(core._embedding._cache, SqliteCache)
        assert core._embedding_cache is core._embedding._cache
    finally:
        asyncio.run(core.close())


def test_rag_core_rejects_sqlite_embedding_cache_without_path() -> None:
    from rag_core import RAGCore
    from rag_core.config import IngestConfig
    from tests.support import (
        FakeEmbeddingProvider,
        FakeSparseEmbedder,
        RecordingVectorStore,
        make_test_config,
    )

    base = make_test_config(
        qdrant_collection="rag_core_cache_provider_test",
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
            enable_lexical_search=base.ingest.enable_lexical_search,
            manifest_directory=base.ingest.manifest_directory,
            embedding_cache_provider=SQLITE_CACHE_PROVIDER,
        ),
        policy=base.policy,
    )

    with pytest.raises(ValueError, match="embedding_cache_path"):
        RAGCore(
            config,
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=RecordingVectorStore(),
        )


def test_rag_core_close_closes_cache_resources(tmp_path: Path) -> None:
    from rag_core import RAGCore
    from tests.support import (
        FakeEmbeddingProvider,
        FakeSparseEmbedder,
        RecordingVectorStore,
        make_test_config,
    )

    embedding_cache = SqliteCache(tmp_path / "embeddings.sqlite")
    chunk_context_cache = SqliteChunkContextCache(tmp_path / "contexts.sqlite")
    asyncio.run(embedding_cache.put(_embed_key(), [0.1, 0.2, 0.3, 0.4]))
    asyncio.run(
        chunk_context_cache.put(
            ChunkContextKey(
                contextualizer_id="ctx",
                document_sha256="doc",
                document_filename_sha256="name",
                chunk_text_sha256="chunk",
                chunk_index=0,
                total_chunks=1,
            ),
            "context",
        )
    )
    assert getattr(embedding_cache, "_connection") is not None
    assert getattr(chunk_context_cache, "_connection") is not None

    core = RAGCore(
        make_test_config(embedding_dimensions=4),
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=RecordingVectorStore(),
        embedding_cache=embedding_cache,
        chunk_context_cache=chunk_context_cache,
    )

    asyncio.run(core.close())

    assert getattr(embedding_cache, "_connection") is None
    assert getattr(chunk_context_cache, "_connection") is None
