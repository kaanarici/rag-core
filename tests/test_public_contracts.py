import ast
import asyncio
import importlib
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, cast

import rag_core
import rag_core.config as config_module
import rag_core.documents as documents_module
import rag_core.documents.converters as converters_module
import rag_core.search as search_module
import rag_core.integrations as integrations
import rag_core.search.providers as provider_exports
import rag_core.search.providers.chunk_context_cache as chunk_context_cache_module
import rag_core.search.providers.embedding_cache as embedding_cache_module
from rag_core import (
    ContextSnippet,
    CorpusManifestEntry,
    CorpusManifest,
    DeleteDocumentResult,
    IngestedDocument,
    ContextPack,
    OcrMetadata,
    OcrRoutingSignal,
    ParsedDocument,
    PreparedChunk,
    PreparedDocument,
    ProcessingFingerprint,
    RAGCore,
    RAGCoreConfig,
    Rag,
    SearchResult,
    SourceLocator,
    SourcePreview,
    SourceReference,
    index,
)
from rag_core.integrations.langchain import (
    LangChainNotInstalledError,
    LangChainRetrieverConfig,
    build_langchain_retriever,
    create_langchain_context_tool,
    create_langchain_retriever_tool,
)
from rag_core.integrations.mcp_server import build_mcp_server
from rag_core.integrations.openai_agents import build_retrieve_context_tool
from rag_core.search.context_pack import (
    ContextSnippet as SearchContextSnippet,
    SourceLocator as SearchSourceLocator,
    SourcePreview as SearchSourcePreview,
    SourceReference as SearchSourceReference,
)
from rag_core.manifest_reconciliation import ManifestSource, reconcile_entries
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


ROOT_DAY_ONE_EXPORTS = (
    "index",
    "Rag",
    "RAGCore",
    "RAGCoreConfig",
    "ContextPack",
    "SearchResult",
)
ROOT_ADVANCED_DOOR_NAMES = (
    "ContextSnippet",
    "CorpusManifest",
    "CorpusManifestEntry",
    "DeleteDocumentResult",
    "IngestedDocument",
    "OcrMetadata",
    "OcrRoutingSignal",
    "ParsedDocument",
    "PreparedChunk",
    "PreparedDocument",
    "ProcessingFingerprint",
    "SourceLocator",
    "SourcePreview",
    "SourceReference",
)


def _stub_public_names(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.ImportFrom):
            names.update(
                alias.asname
                for alias in node.names
                if alias.asname and not alias.asname.startswith("_")
            )
        elif isinstance(node, ast.FunctionDef):
            names.add(node.name)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id != "__all__"
        ):
            names.add(node.target.id)
    return names


def _stub_all_names(path: Path) -> set[str]:
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            ):
                value = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__all__"
        ):
            value = node.value
        if value is None:
            continue
        parsed = ast.literal_eval(value)
        assert isinstance(parsed, (list, tuple))
        return set(parsed)
    raise AssertionError(f"{path} does not define literal __all__")


def test_shipped_public_stubs_match_runtime_exports() -> None:
    stubbed_modules = {
        "rag_core": Path("src/rag_core/__init__.pyi"),
        "rag_core.config": Path("src/rag_core/config/__init__.pyi"),
        "rag_core.contracts": Path("src/rag_core/contracts/__init__.pyi"),
        "rag_core.events": Path("src/rag_core/events/__init__.pyi"),
        "rag_core.integrations": Path("src/rag_core/integrations/__init__.pyi"),
        "rag_core.search": Path("src/rag_core/search/__init__.pyi"),
        "rag_core.search.providers": Path("src/rag_core/search/providers/__init__.pyi"),
    }

    for module_name, stub_path in stubbed_modules.items():
        module = importlib.import_module(module_name)
        assert _stub_all_names(stub_path) == set(module.__all__)
        if module_name == "rag_core":
            stub_names = _stub_public_names(stub_path)
            assert set(ROOT_DAY_ONE_EXPORTS).issubset(stub_names)
            assert set(ROOT_ADVANCED_DOOR_NAMES).issubset(stub_names)
        else:
            assert _stub_public_names(stub_path) == set(module.__all__)


def test_ingest_results_aggregator_exposes_local_and_remote_outcomes() -> None:
    import rag_core.ingest_results as ingest_results
    from rag_core.local_ingest_models import LocalIngestResult as _LocalResult
    from rag_core.remote_ingest_results import (
        RemoteUrlIngestResult as _RemoteUrlResult,
    )

    assert ingest_results.LocalIngestResult is _LocalResult
    assert ingest_results.RemoteUrlIngestResult is _RemoteUrlResult
    assert set(ingest_results.__all__) == {
        "LocalIngestFailure",
        "LocalIngestPlan",
        "LocalIngestRequest",
        "LocalIngestResult",
        "LocalIngestSuccess",
        "LocalManifestStatus",
        "RemoteManifestStatus",
        "RemoteUrlIngestFailure",
        "RemoteUrlIngestResult",
        "RemoteUrlIngestSuccess",
    }
    stability = Path("docs-site/content/docs/stability.mdx").read_text(encoding="utf-8")
    assert "`rag_core.ingest_results`" in stability


def test_shared_limit_bounds_validator_is_used_by_runtime_and_contracts() -> None:
    runtime_source = Path("src/rag_core/runtime/requests.py").read_text(
        encoding="utf-8"
    )
    contracts_source = Path(
        "src/rag_core/contracts/tool_contract_requests.py"
    ).read_text(encoding="utf-8")

    # Both call sites use the shared helper, so caps stay in lock-step.
    assert "validate_limit_bounds" in runtime_source
    assert "from rag_core.contracts.tool_contract_requests" in runtime_source
    assert "def validate_limit_bounds" in contracts_source
    assert "validate_limit_bounds(" in contracts_source

    # And the runtime path enforces the shared LIMIT_MAX (not just >= 1).
    import rag_core.runtime.requests as runtime_requests
    from rag_core.contracts.tool_contract_schemas import (
        SEARCH_USER_DOCUMENTS_LIMIT_MAX,
    )
    from rag_core.runtime.errors import RuntimeRequestError

    over = SEARCH_USER_DOCUMENTS_LIMIT_MAX + 1
    try:
        runtime_requests.parse_retrieval_request(
            {
                "query": "q",
                "namespace": "ns",
                "corpus_ids": ["c"],
                "limit": over,
            }
        )
    except RuntimeRequestError as exc:
        assert "limit must be between" in str(exc)
    else:  # pragma: no cover - assertion failure path
        raise AssertionError("runtime did not enforce shared limit cap")


def test_root_day_one_surface_is_small_and_advanced_shapes_remain_importable() -> None:
    day_one_types = (
        index,
        Rag,
        RAGCore,
        RAGCoreConfig,
        ContextPack,
        SearchResult,
    )
    for value in day_one_types:
        assert getattr(value, "__name__", "") or callable(value)
    assert tuple(rag_core.__all__) == ROOT_DAY_ONE_EXPORTS

    advanced_shapes = (
        ContextSnippet,
        CorpusManifest,
        CorpusManifestEntry,
        DeleteDocumentResult,
        IngestedDocument,
        OcrMetadata,
        OcrRoutingSignal,
        ParsedDocument,
        PreparedChunk,
        PreparedDocument,
        ProcessingFingerprint,
        SourceLocator,
        SourcePreview,
        SourceReference,
    )
    for cls in advanced_shapes:
        # The door: each advanced shape is reachable via ``rag_core.<Name>``
        # (lazy __getattr__) even though it is off the day-one ``__all__``.
        assert getattr(rag_core, cls.__name__) is cls
    assert set(ROOT_ADVANCED_DOOR_NAMES).isdisjoint(set(rag_core.__all__))
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


def test_rag_core_facade_methods_are_curated() -> None:
    beta_core_methods = {
        "close",
        "delete_document",
        "ingest_archive",
        "ingest_bytes",
        "ingest_file",
        "ingest_files",
        "ingest_url",
        "ingest_urls",
        "parse_bytes",
        "prepare_bytes",
        "prepare_file",
        "retrieve_context",
        "search",
    }
    experimental_facade_methods = {
        "build_corpus_manifest",
        "build_manifest_entry",
        "check_health",
        "describe_runtime",
        "ensure_ready",
        "manifest_bytes",
        "manifest_file",
    }
    for method in beta_core_methods | experimental_facade_methods:
        assert callable(getattr(RAGCore, method))
    assert not hasattr(RAGCore, "ingest_folder")


def test_stability_docs_name_the_actual_rag_core_ingest_shape() -> None:
    stability = Path("docs-site/content/docs/stability.mdx").read_text(encoding="utf-8")

    assert "local file/directory paths via `ingest_files`" in stability
    assert "files/folders" not in stability
    assert "`RAGCore.ingest_folder`" not in stability
    assert "`rag_core.search.provider_protocols` protocols" in stability
    # The shim exists for compatibility, but docs should steer new app code to
    # curated or owner-module imports instead of advertising the legacy path.
    assert "rag_core.search.types" not in stability


def test_context_pack_citation_primitives_are_public_imports() -> None:
    assert rag_core.ContextSnippet is SearchContextSnippet
    assert rag_core.SourceLocator is SearchSourceLocator
    assert rag_core.SourcePreview is SearchSourcePreview
    assert rag_core.SourceReference is SearchSourceReference

    assert search_module.ContextSnippet is SearchContextSnippet
    assert search_module.SourceLocator is SearchSourceLocator
    assert search_module.SourcePreview is SearchSourcePreview
    assert search_module.SourceReference is SearchSourceReference


def test_config_namespace_exports_curated_config_shapes_and_constants() -> None:
    assert list(config_module.__all__) == [
        "ChunkingConfig",
        "BUILTIN_CHUNKING_STRATEGIES",
        "CHUNKING_STRATEGY_AUTO",
        "CLI_MANIFEST_DIR_ENV",
        "CODE_CHUNKING_STRATEGY",
        "CONTENT_CHUNKER_CHUNKING_STRATEGY",
        "MARKDOWN_CHUNKING_STRATEGY",
        "PRECHUNKED_CHUNKING_STRATEGY",
        "PUBLIC_CHUNKING_STRATEGIES",
        "SEMANTIC_CHUNKING_STRATEGY",
        "ChunkingStrategyName",
        "DEFAULT_TURBOPUFFER_DELETE_CONTINUATION_LIMIT",
        "DEFAULT_PGVECTOR_SCHEMA",
        "DEFAULT_VECTOR_STORE_PROVIDER",
        "DEFAULT_EMBEDDING_MODEL",
        "DEFAULT_EMBEDDING_PROVIDER",
        "DEFAULT_CLI_MANIFEST_DIRECTORY",
        "DEFAULT_INGEST_MAX_CONCURRENCY",
        "DEFAULT_INGEST_SOURCE_TYPE",
        "DEFAULT_PROCESSING_VERSION",
        "DEFAULT_TURBOPUFFER_DISTANCE_METRIC",
        "DEFAULT_QDRANT_COLLECTION",
        "DEFAULT_QDRANT_DIMENSION_AWARE_COLLECTION",
        "DEFAULT_RERANKER_PROVIDER",
        "DEMO_EMBEDDING_MODEL",
        "DEMO_EMBEDDING_PROVIDER",
        "EMBEDDING_BATCH_SIZE_ENV",
        "EMBEDDING_DIMENSIONS_ENV",
        "EMBEDDING_MODEL_ENV",
        "EMBEDDING_PROVIDER_ENV",
        "EmbeddingConfig",
        "ContextualizerConfig",
        "LOCAL_EMBEDDING_DIMENSIONS",
        "LOCAL_EMBEDDING_MODEL",
        "LOCAL_EMBEDDING_PROVIDER",
        "IngestConfig",
        "INGEST_SOURCE_TYPE_ARCHIVE",
        "INGEST_SOURCE_TYPE_FILE",
        "INGEST_SOURCE_TYPE_URL",
        "PGVECTOR_DSN_ENV",
        "PGVECTOR_SCHEMA_ENV",
        "PGVECTOR_TABLE_ENV",
        "PgVectorStoreConfig",
        "PROCESSING_VERSION_ENV",
        "SKIP_UNCHANGED_FAST",
        "SKIP_UNCHANGED_MATERIALIZE",
        "SkipUnchangedMode",
        "QdrantConfig",
        "QDRANT_COLLECTION_ENV",
        "QDRANT_DIMENSION_AWARE_COLLECTION_ENV",
        "QDRANT_LOCATION_ENV",
        "QDRANT_URL_ENV",
        "RERANKER_MODEL_ENV",
        "RERANKER_PROVIDER_ENV",
        "RerankerConfig",
        "SUPPORTED_TURBOPUFFER_DISTANCE_METRICS",
        "SUPPORTED_VECTOR_STORE_PROVIDERS",
        "STANDARD_INGEST_SOURCE_TYPES",
        "TURBOPUFFER_BASE_URL_ENV",
        "TURBOPUFFER_DELETE_CONTINUATION_LIMIT_ENV",
        "TURBOPUFFER_DISTANCE_METRIC_ENV",
        "TURBOPUFFER_NAMESPACE_ENV",
        "TURBOPUFFER_REGION_ENV",
        "TurboPufferVectorStoreConfig",
        "VECTOR_STORE_ENV",
        "VectorStoreConfig",
    ]
    for helper in (
        "get_env",
        "get_env_bool",
        "get_env_float",
        "get_env_int",
        "get_env_optional",
        "get_env_optional_bool",
        "get_env_stripped",
    ):
        assert not hasattr(config_module, helper)


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


def test_source_reconciliation_primitives_are_app_owned_status_only() -> None:
    entry = CorpusManifestEntry(
        document_id="doc-old",
        namespace="acme",
        corpus_id="help",
        document_key="old.md",
        content_sha256="old-hash",
        filename="old.md",
        mime_type="text/markdown",
        chunk_count=1,
    )

    reconciliation = reconcile_entries(
        [entry],
        [ManifestSource(document_key="new.md", content_sha256="new-hash")],
    )

    assert [(item.status, item.document_key) for item in reconciliation.items] == [
        ("missing", "new.md"),
        ("orphaned", "old.md"),
    ]

def test_integration_root_exports_curated_builders() -> None:
    assert integrations.__all__ == (
        "LangChainNotInstalledError",
        "LangChainRetrieverConfig",
        "build_langchain_retriever",
        "build_mcp_server",
        "build_retrieve_context_tool",
        "create_langchain_context_tool",
        "create_langchain_retriever_tool",
        "langchain",
        "mcp_server",
        "openai_agents",
    )
    assert integrations.LangChainNotInstalledError is LangChainNotInstalledError
    assert integrations.LangChainRetrieverConfig is LangChainRetrieverConfig
    assert integrations.build_langchain_retriever is build_langchain_retriever
    assert integrations.build_mcp_server is build_mcp_server
    assert integrations.build_retrieve_context_tool is build_retrieve_context_tool
    assert integrations.create_langchain_context_tool is create_langchain_context_tool
    assert (
        integrations.create_langchain_retriever_tool is create_langchain_retriever_tool
    )


def test_integration_root_keeps_payload_helpers_in_submodules() -> None:
    assert not hasattr(integrations, "context_pack_to_tool_output")
    assert not hasattr(integrations, "search_result_to_document_kwargs")


def test_wheel_smoke_exercises_installed_integration_public_surface() -> None:
    source = Path("scripts/wheel_smoke.py").read_text(encoding="utf-8")

    assert "import rag_core.integrations as integrations" in source
    assert "def _integration_import_smoke()" in source
    assert "integrations.__all__" in source
    assert "integrations.build_langchain_retriever(" in source
    assert "integrations.build_mcp_server(" in source
    assert "integrations.build_retrieve_context_tool(" in source
    assert "LangChainNotInstalledError" in source
    assert "mcp" in source
    assert "openai-agents" in source


def test_wheel_smoke_exercises_installed_cli_first_run_surface() -> None:
    source = Path("scripts/wheel_smoke.py").read_text(encoding="utf-8")

    assert "def _installed_cli_smoke(" in source
    assert '"doctor", "--json"' in source
    assert '"local-search"' in source
    assert '"Local files parsed indexed cited"' in source
    assert '"installed_cli_local_search_hits"' in source


def test_integration_surfaces_type_against_root_rag_core() -> None:
    for path in (
        Path("src/rag_core/integrations/__init__.pyi"),
        Path("src/rag_core/integrations/langchain.py"),
        Path("src/rag_core/integrations/langchain_retriever.py"),
    ):
        source = path.read_text(encoding="utf-8")

        assert "from rag_core import RAGCore" in source
        assert "from rag_core.core import RAGCore" not in source


def test_eval_runner_types_against_root_rag_core() -> None:
    source = Path("src/rag_core/evals/runner.py").read_text(encoding="utf-8")

    assert "from rag_core import RAGCore" in source
    assert "from rag_core.search import QueryPlan, RerankBudget, SearchResult" in source
    assert "from rag_core.core import RAGCore" not in source


def test_root_package_exports_search_types_through_curated_search_surface() -> None:
    root = Path("src/rag_core/__init__.py").read_text(encoding="utf-8")
    stub = Path("src/rag_core/__init__.pyi").read_text(encoding="utf-8")

    for source in (root, stub):
        assert "from .search import ContextPack" in source
        assert "from .search import SearchResult" in source
        assert ".search.context_pack import" not in source
        assert ".search.types import SearchResult" not in source
    assert '"SearchResult": ("rag_core.search", "SearchResult")' in root
    assert '"ContextPack": ("rag_core.search", "ContextPack")' in root
    assert '"rag_core.search.types", "SearchResult"' not in root


def test_public_extension_modules_type_against_curated_search_surface() -> None:
    for path in (
        Path("src/rag_core/evals/runner.py"),
        Path("src/rag_core/events/export.py"),
        Path("src/rag_core/integrations/protocols.py"),
        Path("src/rag_core/integrations/openai_agents.py"),
        Path("src/rag_core/integrations/mcp_server.py"),
        Path("src/rag_core/integrations/langchain_payloads.py"),
        Path("src/rag_core/integrations/langchain_retriever.py"),
    ):
        source = path.read_text(encoding="utf-8")

        assert "rag_core.search.types" not in source
        assert "rag_core.search.query_plan" not in source
        assert "rag_core.search.vector_models" not in source


def test_user_facing_retrieval_modules_type_against_curated_search_surface() -> None:
    for path in (
        Path("src/rag_core/_engine/core_retrieval.py"),
        Path("src/rag_core/facade/retrieval.py"),
        Path("src/rag_core/cli_search.py"),
        Path("src/rag_core/local_search_runner.py"),
    ):
        source = path.read_text(encoding="utf-8")

        assert "from rag_core.search import" in source
        assert "from rag_core.search.types import" not in source
        assert "from rag_core.search.query_plan import" not in source
        assert "from rag_core.search.context_pack_models import" not in source
        assert "rag_core.search.vector_models" not in source


def test_integration_package_stub_uses_single_retrieve_context_protocol() -> None:
    source = Path("src/rag_core/integrations/__init__.pyi").read_text(encoding="utf-8")
    mcp_stub = Path("src/rag_core/integrations/mcp_server.pyi").read_text(
        encoding="utf-8"
    )

    assert "class SupportsRetrieveContext" not in source
    assert "import rag_core.integrations.protocols" in source
    assert "core: rag_core.integrations.protocols.SupportsRetrieveContext" in source
    assert "SupportsSearchAndRetrieveContext" not in source
    assert "SupportsSearchAndRetrieveContext" not in mcp_stub
    assert "core: SupportsRetrieveContext" in mcp_stub


def test_integration_protocols_reuse_tool_contract_context_pack_protocol() -> None:
    protocols = Path("src/rag_core/integrations/protocols.py").read_text(
        encoding="utf-8"
    )
    context_text = Path("src/rag_core/integrations/integration_context_text.py")
    langchain_payloads = Path(
        "src/rag_core/integrations/langchain_payloads.py"
    ).read_text(encoding="utf-8")
    langchain_stub = Path("src/rag_core/integrations/langchain.pyi").read_text(
        encoding="utf-8"
    )
    tool_contracts = Path("src/rag_core/contracts/tool_contracts.py").read_text(
        encoding="utf-8"
    )

    assert "class ContextPackLike" not in protocols
    assert "SupportsContextPackPromptPayload" in protocols
    assert not context_text.exists()
    assert "SupportsContextPackPromptPayload" in langchain_payloads
    assert "pack: SupportsContextPackPromptPayload" in langchain_stub
    assert "class SupportsContextPackPayload" not in tool_contracts
    assert (
        "class SupportsContextPackPromptPayload(SupportsContextPackPayload"
        not in tool_contracts
    )


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
        "ContextPack",
        "Not",
        "Or",
        "Prefetch",
        "PrefetchFusion",
        "PRIMARY_DENSE_QUERY_VECTOR",
        "QUERY_PLAN_PRESETS",
        "QueryPlan",
        "Range",
        "RerankBudget",
        "SEARCH_PROFILES",
        "SearchResult",
        "SparseChannel",
        "SparseVector",
        "SourceLocator",
        "SourcePreview",
        "SourceReference",
        "Term",
        "UnsupportedQueryStage",
        "default_query_plan",
        "describe_query_plan",
        "describe_query_plan_presets",
        "describe_search_profile_catalog",
        "describe_search_profiles",
        "query_plan_preset",
        "search_profile",
    )
    # Sidecar types are intentionally private. Guard against accidental re-export.
    search_exports = cast(Any, search_module)
    assert not hasattr(search_exports, "FuseStage")
    assert not hasattr(search_exports, "PipelineContext")
    assert not hasattr(search_exports, "ProviderRerankStage")
    assert not hasattr(search_exports, "QdrantIndexer")
    assert not hasattr(search_exports, "MetadataFilterCapabilities")
    assert not hasattr(search_exports, "SearchExecutionOptions")
    assert not hasattr(search_exports, "SearchRequest")
    assert not hasattr(search_exports, "SearchQuery")
    assert not hasattr(search_exports, "SearchPipelineRunner")
    assert not hasattr(search_exports, "StoreCapabilities")
    assert not hasattr(search_exports, "VectorStoreProviderSpec")
    assert not hasattr(search_exports, "SidecarPrefetchTransform")
    assert not hasattr(search_exports, "build_context_pack")
    assert not hasattr(search_exports, "PortableLexicalSidecar")
    assert not hasattr(search_exports, "LexicalSidecarRecord")


def test_provider_exports_are_curated() -> None:
    provider_stub = Path("src/rag_core/search/providers/__init__.pyi").read_text(
        encoding="utf-8"
    )
    assert provider_exports.__all__ == (
        "CHUNK_CONTEXT_CACHES",
        "ChunkContextCache",
        "ChunkContextKey",
        "EMBEDDING_CACHES",
        "EMBEDDING_PROVIDERS",
        "EmbedCacheKey",
        "EmbeddingCache",
        "ProviderRegistry",
        "PgVectorVectorStore",
        "QdrantVectorStore",
        "RERANKER_PROVIDERS",
        "SEARCH_SIDECARS",
        "SPARSE_EMBEDDERS",
        "VECTOR_STORES",
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
        "LocalEmbeddingProvider",
        "NoCache",
        "NoChunkContextCache",
        "OpenAIEmbeddingProvider",
        "QueryPlanCapabilities",
        "RichVectorStore",
        "SqliteCache",
        "SqliteChunkContextCache",
        "StoreCapabilities",
        "TurboPufferVectorStore",
        "VectorStorePolicy",
    ):
        assert hidden not in provider_exports.__all__
        assert not hasattr(provider_exports, hidden)
        assert f"{hidden} as {hidden}" not in provider_stub
    for annotation_only_protocol in (
        "EmbeddingProvider",
        "RerankerProvider",
        "SearchSidecar",
        "SparseEmbedder",
    ):
        assert "from rag_core.search.provider_protocols import" in provider_stub
        assert f"{annotation_only_protocol} as _{annotation_only_protocol}" in (
            provider_stub
        )
        assert annotation_only_protocol not in provider_exports.__all__
        assert not hasattr(provider_exports, annotation_only_protocol)
    # EmbedCacheKey shape is part of the cache contract. Preserve order.
    # The trailing ``namespace`` / ``corpus_id`` / ``document_id`` fields
    # carry the right-to-forget scope so a per-document delete actually
    # purges scoped rows in the cache.
    assert [field.name for field in fields(provider_exports.EmbedCacheKey)] == [
        "provider",
        "provider_config_fingerprint",
        "model",
        "dimensions",
        "input_type",
        "normalization",
        "processing_fingerprint",
        "content_sha256",
        "namespace",
        "corpus_id",
        "document_id",
    ]


def test_provider_cache_contract_exports_use_canonical_modules() -> None:
    provider_root = Path("src/rag_core/search/providers/__init__.py").read_text(
        encoding="utf-8"
    )
    provider_stub = Path("src/rag_core/search/providers/__init__.pyi").read_text(
        encoding="utf-8"
    )

    assert "rag_core.search.providers.chunk_context_cache" in provider_root
    assert "rag_core.search.providers.embedding_cache_models" in provider_root
    assert '"create_chunk_context_cache": (' in provider_root
    assert "from rag_core.search.providers.chunk_context_cache import" in provider_stub
    assert (
        "from rag_core.search.providers.embedding_cache_models import" in provider_stub
    )
    assert "create_chunk_context_cache" in chunk_context_cache_module.__all__
    assert "create_chunk_context_cache" not in embedding_cache_module.__all__
    assert "EmbeddingCache" not in embedding_cache_module.__all__
    assert "EmbedCacheKey" not in embedding_cache_module.__all__
    assert "ChunkContextCache" not in embedding_cache_module.__all__
    assert "ChunkContextKey" not in embedding_cache_module.__all__
    assert "InMemoryChunkContextCache" not in embedding_cache_module.__all__
    assert "NoChunkContextCache" not in embedding_cache_module.__all__
    assert "SqliteChunkContextCache" not in embedding_cache_module.__all__
    assert "sha256_text" not in embedding_cache_module.__all__


def test_stability_docs_do_not_overclaim_provider_root_vector_store_exports() -> None:
    stability = Path("docs-site/content/docs/stability.mdx").read_text(encoding="utf-8")

    assert "the default Qdrant vector-store adapter" in stability
    assert "optional/utility vector stores stay in their owning modules" in stability
    assert "vector store adapters" not in stability


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


def test_documents_exports_are_curated() -> None:
    assert documents_module.__all__ == (
        "AnthropicChunkContextualizer",
        "CHUNKING_STRATEGIES",
        "CachingContextualizer",
        "ChunkContextRequest",
        "ChunkContextualizer",
        "CommandOcrProvider",
        "CONTEXTUALIZER_PROVIDERS",
        "LocalParseError",
        "NoOpContextualizer",
        "OCR_PROVIDERS",
        "OcrProvider",
        "OcrRequest",
        "OcrResult",
        "build_gemini_ocr_provider",
        "build_mistral_ocr_provider",
        "create_chunking_strategy",
        "create_contextualizer",
        "create_ocr_provider",
        "parse_file_bytes",
    )
    assert not hasattr(documents_module, "PdfConverter")
    assert not hasattr(documents_module, "PdfInspectorDetectionResult")


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


def test_engine_implementation_modules_are_private_package_files() -> None:
    root_core_modules = sorted(
        path.name
        for path in Path("src/rag_core").glob("core_*.py")
        if path.name != "core_models.py"
    )
    private_engine_modules = sorted(
        path.name for path in Path("src/rag_core/_engine").glob("core_*.py")
    )

    assert root_core_modules == []
    assert "core_ingest.py" in private_engine_modules
    assert "core_prepare.py" in private_engine_modules
    assert "core_retrieval.py" in private_engine_modules
