import asyncio
import subprocess
import sys
from dataclasses import fields
from typing import Any, cast

import rag_core
import rag_core.documents.converters as converters_module
import rag_core.search as search_module
import rag_core.integrations as integrations
import rag_core.search as search
import rag_core.search.providers as provider_exports
from rag_core import (
    CorpusManifestEntry,
    IngestedDocument,
    ModelContextPack,
    ParsedDocument,
    PreparedChunk,
    PreparedDocument,
    RAGCore,
    RAGCoreConfig,
    SearchResult,
)
from rag_core import sources
from rag_core.integrations.langchain import (
    LangChainNotInstalledError,
    LangChainRetrieverConfig,
    build_langchain_retriever,
    create_langchain_context_tool,
    create_langchain_retriever_tool,
)
from rag_core.integrations.openai_agents import build_retrieve_context_tool
from rag_core.search.context_pack import (
    ContextSnippet,
    SourceLocator,
    SourcePreview,
    SourceReference,
)
from rag_core.sources import (
    ArchiveLimits,
    ArchiveSourceItem,
    ArchiveSourcePlan,
    LocalFileSourceReader,
    LocalSourceItem,
    LocalSourcePlan,
    RemoteDiscoveredUrl,
    RemoteDiscovery,
    RemoteDiscoveryReader,
    RemoteSourceDocument,
    RemoteUrlSourceReader,
    ZipArchiveSourceReader,
    parse_llms_txt_urls,
    parse_sitemap_urls,
)
from rag_core.search.providers.cached_embedding import (
    CachedEmbeddingDiagnostics,
    EmbeddingCacheObservation,
)

from tests.support import (
    FakeEmbeddingProvider,
    FakeSparseEmbedder,
    RecordingVectorStore,
    make_search_result,
    make_test_config,
)


def test_root_public_types_are_importable() -> None:
    # The root package stays small: engine facade plus first-order data shapes.
    public_types = (
        CorpusManifestEntry,
        RAGCore,
        RAGCoreConfig,
        ParsedDocument,
        IngestedDocument,
        PreparedChunk,
        PreparedDocument,
        ModelContextPack,
        SearchResult,
    )
    for cls in public_types:
        assert cls.__name__ == cls.__name__
    assert tuple(rag_core.__all__) == (
        "ContextSnippet",
        "CorpusManifest",
        "CorpusManifestEntry",
        "IngestedDocument",
        "ModelContextPack",
        "OcrMetadata",
        "OcrRoutingSignal",
        "ParsedDocument",
        "PreparedChunk",
        "PreparedDocument",
        "ProcessingFingerprint",
        "RAGCore",
        "RAGCoreConfig",
        "SearchResult",
        "SourceLocator",
        "SourcePreview",
        "SourceReference",
    )
    assert {
        "build_context_pack",
        "EventBuffer",
        "EvalCase",
        "FetchSecurityPolicy",
        "LocalFileSourceReader",
        "CachedEmbeddingProvider",
    }.isdisjoint(set(rag_core.__all__))
    assert set(rag_core.__all__).issubset(dir(rag_core))
    assert CachedEmbeddingDiagnostics.__name__ == "CachedEmbeddingDiagnostics"
    assert EmbeddingCacheObservation.__name__ == "EmbeddingCacheObservation"


def test_context_pack_citation_primitives_are_public_imports() -> None:
    assert rag_core.ContextSnippet is ContextSnippet
    assert rag_core.SourceLocator is SourceLocator
    assert rag_core.SourcePreview is SourcePreview
    assert rag_core.SourceReference is SourceReference

    assert search.ContextSnippet is ContextSnippet
    assert search.SourceLocator is SourceLocator
    assert search.SourcePreview is SourcePreview
    assert search.SourceReference is SourceReference


def test_source_preview_remains_a_small_app_facing_payload() -> None:
    preview = SourcePreview(
        citation_id="billing#chunk-0",
        title="billing.md",
        locator_label="page 2, chunk 0",
        document_id="billing",
        corpus_id="help",
        source_hash="sha256:abc",
    )

    assert preview.as_text() == "[billing#chunk-0] billing.md (page 2, chunk 0)"
    assert preview.to_payload()["source_hash"] == "sha256:abc"


def test_source_primitives_are_sources_namespace_public_imports() -> None:
    assert ArchiveLimits.__name__ == "ArchiveLimits"
    assert ArchiveSourceItem.__name__ == "ArchiveSourceItem"
    assert ArchiveSourcePlan.__name__ == "ArchiveSourcePlan"
    assert LocalFileSourceReader.__name__ == "LocalFileSourceReader"
    assert LocalSourceItem.__name__ == "LocalSourceItem"
    assert LocalSourcePlan.__name__ == "LocalSourcePlan"
    assert RemoteDiscoveredUrl.__name__ == "RemoteDiscoveredUrl"
    assert RemoteDiscovery.__name__ == "RemoteDiscovery"
    assert RemoteDiscoveryReader.__name__ == "RemoteDiscoveryReader"
    assert RemoteSourceDocument.__name__ == "RemoteSourceDocument"
    assert RemoteUrlSourceReader.__name__ == "RemoteUrlSourceReader"
    assert ZipArchiveSourceReader.__name__ == "ZipArchiveSourceReader"
    assert callable(parse_llms_txt_urls)
    assert callable(parse_sitemap_urls)


def test_remote_discovery_exposes_fetchable_and_redacted_url_sequences() -> None:
    discovery = parse_llms_txt_urls(
        "- [Guide](/guide?private=alpha)\n",
        base_url="https://example.com/llms.txt",
    )

    assert discovery.urls == ("https://example.com/guide?private=alpha",)
    assert discovery.redacted_urls == ("https://example.com/guide?redacted",)
    assert "private=alpha" not in repr(discovery)


def test_sources_namespace_is_curated() -> None:
    assert sources.__all__ == [
        "ArchiveLimits",
        "ArchiveSourceItem",
        "ArchiveSourcePlan",
        "LocalFileSourceReader",
        "LocalSourceItem",
        "LocalSourcePlan",
        "RemoteDiscoveredUrl",
        "RemoteDiscovery",
        "RemoteDiscoveryKind",
        "RemoteDiscoveryReader",
        "RemoteSourceDocument",
        "RemoteUrlSourceReader",
        "ZipArchiveSourceReader",
        "archive_document_key",
        "document_key",
        "expand_supported_local_files",
        "file_content_sha256",
        "is_supported_archive_member_path",
        "is_ignored_local_file",
        "is_supported_local_candidate",
        "is_supported_local_file",
        "local_file_source_item",
        "local_source_key_root",
        "parse_llms_txt_urls",
        "parse_sitemap_urls",
        "read_zip_member_bytes",
        "remote_source_document",
        "safe_archive_member_path",
        "source_error_message",
        "write_discovered_url_file",
        "write_raw_discovered_url_file",
    ]


def test_integration_root_exports_stable_builders() -> None:
    assert integrations.__all__ == (
        "LangChainNotInstalledError",
        "LangChainRetrieverConfig",
        "build_langchain_retriever",
        "build_retrieve_context_tool",
        "create_langchain_context_tool",
        "create_langchain_retriever_tool",
        "langchain",
        "openai_agents",
    )
    assert integrations.LangChainNotInstalledError is LangChainNotInstalledError
    assert integrations.LangChainRetrieverConfig is LangChainRetrieverConfig
    assert integrations.build_langchain_retriever is build_langchain_retriever
    assert integrations.build_retrieve_context_tool is build_retrieve_context_tool
    assert integrations.create_langchain_context_tool is create_langchain_context_tool
    assert integrations.create_langchain_retriever_tool is create_langchain_retriever_tool


def test_integration_root_keeps_payload_helpers_in_submodules() -> None:
    assert not hasattr(integrations, "context_pack_to_tool_output")
    assert not hasattr(integrations, "search_result_to_document_kwargs")


def test_search_exports_are_curated() -> None:
    assert search_module.__all__ == (
        "And",
        "Boost",
        "ContextSnippet",
        "DEFAULT_SEARCH_PROFILE",
        "DenseChannel",
        "Filter",
        "Geo",
        "In",
        "Mmr",
        "MetadataFilterCapabilities",
        "ModelContextPack",
        "Not",
        "Or",
        "Prefetch",
        "PrefetchFusion",
        "QUERY_PLAN_PRESETS",
        "QueryPlan",
        "Range",
        "RerankBudget",
        "SEARCH_PROFILES",
        "SearchQuery",
        "SearchRequest",
        "SearchResult",
        "SparseChannel",
        "SparseVector",
        "SourceLocator",
        "SourcePreview",
        "SourceReference",
        "Term",
        "UnsupportedQueryStage",
        "default_query_plan",
        "describe_query_plan_presets",
        "describe_retrieval_profiles",
        "describe_search_profiles",
        "query_plan_preset",
        "search_profile",
    )
    # Sidecar types are intentionally private — guard against accidental re-export.
    search_exports = cast(Any, search_module)
    assert not hasattr(search_exports, "FuseStage")
    assert not hasattr(search_exports, "PipelineContext")
    assert not hasattr(search_exports, "ProviderRerankStage")
    assert not hasattr(search_exports, "QdrantIndexer")
    assert not hasattr(search_exports, "SearchOrchestrator")
    assert not hasattr(search_exports, "SidecarPrefetchTransform")
    assert not hasattr(search_exports, "build_context_pack")
    assert not hasattr(search_exports, "PortableLexicalSidecar")
    assert not hasattr(search_exports, "LexicalSidecarRecord")


def test_provider_exports_are_curated() -> None:
    assert provider_exports.__all__ == (
        "CHUNK_CONTEXT_CACHES",
        "ChunkContextCache",
        "ChunkContextKey",
        "EMBEDDING_CACHES",
        "EMBEDDING_PROVIDERS",
        "EmbedCacheKey",
        "EmbeddingCache",
        "ProviderRegistry",
        "QueryPlanCapabilities",
        "QdrantVectorStore",
        "RERANKER_PROVIDERS",
        "SEARCH_SIDECARS",
        "SPARSE_EMBEDDERS",
        "StoreCapabilities",
        "VECTOR_STORES",
        "VectorStorePolicy",
        "create_chunk_context_cache",
        "create_embedding_cache",
        "create_embedding_provider",
        "create_reranker",
        "create_search_sidecar",
        "create_sparse_embedder",
    )
    # Concrete implementation helpers stay in their modules, not the package root.
    for hidden in (
        "CachedEmbeddingProvider",
        "FastEmbedSparseEmbedder",
        "InMemoryCache",
        "InMemoryChunkContextCache",
        "InMemoryVectorStore",
        "NoCache",
        "NoChunkContextCache",
        "OpenAIEmbeddingProvider",
        "RichVectorStore",
        "SqliteCache",
        "SqliteChunkContextCache",
    ):
        assert hidden not in provider_exports.__all__
        assert not hasattr(provider_exports, hidden)
    # EmbedCacheKey shape is part of the cache contract — preserve order.
    assert [field.name for field in fields(provider_exports.EmbedCacheKey)] == [
        "provider",
        "provider_config_fingerprint",
        "model",
        "dimensions",
        "input_type",
        "normalization",
        "processing_fingerprint",
        "content_sha256",
    ]


def test_converter_exports_are_curated() -> None:
    assert converters_module.__all__ == (
        "BaseConverter",
        "ConversionResult",
        "QualityVerdict",
        "convert_file",
        "get_converter",
    )
    # Concrete converters stay private; consumers go through get_converter/convert_file.
    assert not hasattr(converters_module, "PdfConverter")
    assert not hasattr(converters_module, "TextConverter")


def test_lazy_public_modules_do_not_type_unknown_attributes_as_any() -> None:
    snippets = [
        "import rag_core; reveal_type(rag_core.RAGCor)",
        "import rag_core.search as search; reveal_type(search.QueryPlanX)",
        "import rag_core.events as events; reveal_type(events.NoSuchEvent)",
        "import rag_core.integrations as integrations; reveal_type(integrations.build_retrieve_context_tooool)",
        "import rag_core.integrations as integrations; reveal_type(integrations.openai_agents)",
        "import rag_core.search.providers as providers; reveal_type(providers.create_embedding_providerr)",
    ]
    for snippet in snippets:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mypy",
                "--config-file",
                "/dev/null",
                "-c",
                snippet,
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if "openai_agents" in snippet:
            assert result.returncode == 0
            assert "types.ModuleType" in result.stdout
        else:
            assert result.returncode != 0
            assert "has no attribute" in result.stdout


def test_rag_core_search_returns_public_search_result_with_payload() -> None:
    async def scenario() -> tuple[list[SearchResult], RecordingVectorStore]:
        store = RecordingVectorStore(
            search_results=[
                make_search_result(
                    id="hit-1",
                    text="fox result",
                    score=0.88,
                    document_id="doc-7",
                    corpus_id="corpus-a",
                    section_path="Guide > Retrieval",
                    document_path="/docs/guide.md",
                    chunk_index=2,
                    chunk_word_count=17,
                    chunk_token_estimate=23,
                    embedding_model="fake-embedding",
                    chunker_strategy="markdown",
                    result_type="image",
                    figure_id="figure-1",
                    figure_thumbnail_url="thumb.png",
                    metadata={"team": "search"},
                )
            ]
        )
        core = RAGCore(
            make_test_config(
                qdrant_collection="rag_core_public_contracts", embedding_dimensions=4
            ),
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=store,
        )
        try:
            hits = await core.search(
                query="fox query",
                namespace="team-space",
                corpus_ids=["corpus-a"],
                limit=3,
                document_ids=["doc-7"],
                rerank=False,
            )
        finally:
            await core.close()
        return hits, store

    hits, store = asyncio.run(scenario())
    [hit] = hits
    assert isinstance(hit, SearchResult)
    assert hit.document_id == "doc-7"
    assert hit.section_path == "Guide > Retrieval"
    assert hit.document_path == "/docs/guide.md"
    assert hit.chunk_index == 2
    assert hit.chunk_word_count == 17
    assert hit.chunk_token_estimate == 23
    assert hit.embedding_model == "fake-embedding"
    assert hit.chunker_strategy == "markdown"
    assert hit.result_type == "image"
    assert hit.figure_id == "figure-1"
    assert hit.figure_thumbnail_url == "thumb.png"
    assert hit.metadata == {"team": "search"}

    query = store.search_calls[0]
    assert query.namespace == "team-space"
    assert query.corpus_ids == ["corpus-a"]
    assert query.document_ids == ["doc-7"]
