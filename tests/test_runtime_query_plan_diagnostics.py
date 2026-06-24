from __future__ import annotations

import asyncio
import importlib.machinery
from collections.abc import Callable

from rag_core import Engine
from rag_core.cli.commands.doctor import _planned_core_payload
from rag_core.config import EmbeddingConfig, QdrantConfig
from rag_core.core_models import Config
from rag_core.documents.contextualizer_provider_names import NOOP_CONTEXTUALIZER_ID
from rag_core.documents.contextualizer import NoOpContextualizer
from rag_core.events import EventBuffer, MultiSink
from rag_core.events.sinks import (
    BUFFER_EVENT_SINK_PROVIDER,
    DEFAULT_EVENT_SINK_PROVIDER,
    MULTI_EVENT_SINK_PROVIDER,
)
import rag_core.search.providers.model_provider_specs as diagnostics_module
from rag_core.search.lexical_sidecar import PortableLexicalSidecar
from rag_core.search.providers.cache_sqlite import IN_MEMORY_CACHE_PROVIDER
from rag_core.search.providers.chunk_context_cache import InMemoryChunkContextCache
from rag_core.search.providers.diagnostic_support import (
    MATURITY_OPTIONAL,
    MATURITY_UTILITY,
    MATURITY_INJECTED,
)
from rag_core.search.providers.memory_store import InMemoryVectorStore
from rag_core.search.providers.qdrant_store import QdrantVectorStore
from rag_core.search.providers.vector_store_capabilities import (
    describe_metadata_filter_capabilities,
    describe_query_plan_capabilities,
)
from rag_core.search.providers.vector_store_diagnostics import (
    VECTOR_STORE_QUERY_PLAN_SCOPE_ADAPTER_MAXIMUM,
)

from tests.support import (
    FakeEmbeddingProvider,
    FakeReranker,
    FakeSparseEmbedder,
    make_test_config,
)


def _find_spec_for(
    *available: str,
) -> Callable[[str], importlib.machinery.ModuleSpec | None]:
    available_names = set(available)

    def find_spec(name: str) -> importlib.machinery.ModuleSpec | None:
        if name in available_names:
            return importlib.machinery.ModuleSpec(name, loader=None)
        return None

    return find_spec


def test_describe_runtime_exposes_vector_store_query_plan_capabilities() -> None:
    store = InMemoryVectorStore()
    core = Engine(
        make_test_config(embedding_dimensions=4),
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store,
    )
    try:
        payload = core.describe_runtime()
    finally:
        asyncio.run(core.close())

    vector_store = payload["vector_store"]
    assert isinstance(vector_store, dict)
    capabilities = vector_store["capabilities"]
    assert isinstance(capabilities, dict)
    assert capabilities["per_point_delete"] is True
    assert capabilities["document_record_lookup"] is True
    assert capabilities["query_plan"] == {
        "dense": True,
        "sparse": True,
        "hybrid": True,
        "hybrid_rrf": True,
        "hybrid_dbsf": False,
        "hybrid_weighted_rrf": False,
        "mmr": False,
        "boost": False,
        "nested_prefetch": False,
    }
    assert capabilities["metadata_filter"] == {
        "term": True,
        "in": True,
        "numeric_range": True,
        "string_range": True,
        "geo": True,
        "boolean": True,
    }


def test_describe_runtime_exposes_model_provider_diagnostics(
    monkeypatch,
) -> None:
    monkeypatch.setenv("COHERE_API_KEY", "cohere-secret")
    monkeypatch.setattr(
        diagnostics_module.importlib.util,
        "find_spec",
        _find_spec_for("cohere", "voyageai"),
    )
    embedding = FakeEmbeddingProvider(
        vocabulary=tuple(f"term_{index}" for index in range(512))
    )
    core = Engine(
        make_test_config(
            embedding_provider="voyage",
            embedding_model="voyage-4-lite",
            embedding_dimensions=512,
            embedding_api_key="voyage-secret",
            reranker_provider="cohere",
            reranker_model="rerank-v3.5",
        ),
        embedding_provider=embedding,
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=InMemoryVectorStore(),
        reranker=FakeReranker(),
    )
    try:
        payload = core.describe_runtime()
    finally:
        asyncio.run(core.close())

    providers = payload["providers"]
    assert isinstance(providers, dict)
    embedding_providers = providers["embedding"]
    assert isinstance(embedding_providers, dict)
    embedding_ready = embedding_providers["providers"]
    assert isinstance(embedding_ready, dict)
    voyage = embedding_ready["voyage"]
    assert isinstance(voyage, dict)
    assert voyage["configured"] is True
    assert voyage["maturity"] == MATURITY_OPTIONAL
    assert voyage["package_available"] is True
    assert voyage["api_key_configured"] is True
    assert voyage["dimensions"] == 512

    reranker_providers = providers["reranker"]
    assert isinstance(reranker_providers, dict)
    assert reranker_providers["effective"] == "cohere"
    cohere = reranker_providers["providers"]["cohere"]
    assert isinstance(cohere, dict)
    assert cohere["configured"] is True
    assert cohere["package_available"] is True
    assert cohere["api_key_configured"] is True
    for category in (
        "sparse",
        "ocr",
        "contextualizer",
        "embedding_cache",
        "chunk_context_cache",
        "search_sidecar",
        "event_sink",
    ):
        assert category in providers
        category_payload = providers[category]
        assert isinstance(category_payload, dict)
        assert "providers" in category_payload
    sparse = providers["sparse"]
    assert isinstance(sparse, dict)
    assert sparse["configured"] == "fakesparseembedder"
    assert isinstance(sparse["providers"], dict)
    assert sparse["providers"]["fakesparseembedder"]["maturity"] == MATURITY_INJECTED
    event_sink = providers["event_sink"]
    assert isinstance(event_sink, dict)
    assert event_sink["configured"] == DEFAULT_EVENT_SINK_PROVIDER
    assert "secret" not in repr(payload)


def test_describe_runtime_exposes_demo_embedding_provider_diagnostics() -> None:
    core = Engine(
        make_test_config(
            embedding_provider="demo",
            embedding_model="demo-dense-v1",
            embedding_dimensions=4,
        ),
        vector_store=InMemoryVectorStore(),
    )
    try:
        payload = core.describe_runtime()
    finally:
        asyncio.run(core.close())

    providers = payload["providers"]
    assert isinstance(providers, dict)
    embedding = providers["embedding"]
    assert isinstance(embedding, dict)
    assert embedding["configured"] == "demo"
    demo = embedding["providers"]["demo"]
    assert isinstance(demo, dict)
    assert demo["maturity"] == MATURITY_UTILITY
    assert demo["configured"] is True
    assert demo["package_available"] is True
    assert demo["model"] == "demo-dense-v1"
    assert demo["dimensions"] == 4
    assert "api_key_env" not in demo
    assert "api_key_configured" not in demo


def test_describe_runtime_reports_injected_provider_category_state() -> None:
    sidecar = PortableLexicalSidecar([])
    events = EventBuffer()
    contextualizer = NoOpContextualizer()
    chunk_context_cache = InMemoryChunkContextCache()
    core = Engine(
        make_test_config(embedding_dimensions=4),
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=InMemoryVectorStore(),
        search_sidecar=sidecar,
        event_sink=events,
        chunk_contextualizer=contextualizer,
        chunk_context_cache=chunk_context_cache,
    )
    try:
        payload = core.describe_runtime()
    finally:
        asyncio.run(core.close())

    providers = payload["providers"]
    assert isinstance(providers, dict)
    search_sidecar = providers["search_sidecar"]
    event_sink = providers["event_sink"]
    contextualizer_payload = providers["contextualizer"]
    chunk_context_cache_payload = providers["chunk_context_cache"]
    assert isinstance(search_sidecar, dict)
    assert isinstance(event_sink, dict)
    assert isinstance(contextualizer_payload, dict)
    assert isinstance(chunk_context_cache_payload, dict)
    assert search_sidecar["configured"] == "portable_lexical"
    assert event_sink["configured"] == BUFFER_EVENT_SINK_PROVIDER
    assert contextualizer_payload["configured"] == NOOP_CONTEXTUALIZER_ID
    assert chunk_context_cache_payload["configured"] == IN_MEMORY_CACHE_PROVIDER


def test_describe_runtime_reports_multi_event_sink_as_builtin() -> None:
    events = MultiSink(EventBuffer())
    core = Engine(
        make_test_config(embedding_dimensions=4),
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=InMemoryVectorStore(),
        event_sink=events,
    )
    try:
        payload = core.describe_runtime()
    finally:
        asyncio.run(core.close())

    providers = payload["providers"]
    assert isinstance(providers, dict)
    event_sink = providers["event_sink"]
    assert isinstance(event_sink, dict)
    assert event_sink["configured"] == MULTI_EVENT_SINK_PROVIDER
    event_sink_providers = event_sink["providers"]
    assert isinstance(event_sink_providers, dict)
    multi = event_sink_providers[MULTI_EVENT_SINK_PROVIDER]
    assert isinstance(multi, dict)
    assert multi["configured"] is True
    assert multi["maturity"] == MATURITY_UTILITY


def test_doctor_qdrant_query_plan_diagnostics_match_adapter_capabilities() -> None:
    store = QdrantVectorStore(
        url=None,
        api_key=None,
        collection_name="docs",
        location=":memory:",
        dense_dimensions=4,
    )
    try:
        expected = describe_query_plan_capabilities(store.capabilities.query_plan)
        expected_metadata_filter = describe_metadata_filter_capabilities(
            store.capabilities.metadata_filter
        )
    finally:
        asyncio.run(store.close())

    payload = _planned_core_payload(
        Config(
            qdrant=QdrantConfig(location=":memory:"),
            embedding=EmbeddingConfig(model="text-embedding-3-small", dimensions=4),
        )
    )

    vector_store = payload["vector_store"]
    assert isinstance(vector_store, dict)
    providers = vector_store["providers"]
    assert isinstance(providers, dict)
    qdrant = providers["qdrant"]
    assert isinstance(qdrant, dict)
    assert qdrant["query_plan_scope"] == VECTOR_STORE_QUERY_PLAN_SCOPE_ADAPTER_MAXIMUM
    assert qdrant["per_point_delete"] == store.capabilities.per_point_delete
    assert qdrant["document_record_lookup"] == store.capabilities.document_record_lookup
    assert qdrant["query_plan"] == expected
    assert qdrant["metadata_filter"] == expected_metadata_filter
    memory = providers["memory"]
    assert isinstance(memory, dict)
    assert memory["maturity"] == MATURITY_UTILITY
    assert memory["configured"] is False
    assert memory["check_store_supported"] is False
    assert memory["query_plan"]["hybrid_rrf"] is True
    assert memory["metadata_filter"]["boolean"] is True
