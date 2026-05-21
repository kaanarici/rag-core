from __future__ import annotations

import asyncio
import importlib.machinery
from collections.abc import Callable

from rag_core import RAGCore
from rag_core.cli_doctor import _planned_runtime_payload
from rag_core.config import (
    EmbeddingConfig,
    QdrantConfig,
    TurboPufferVectorStoreConfig,
    VectorStoreConfig,
)
from rag_core.core_models import RAGCoreConfig
from rag_core.core_runtime import describe_query_plan_capabilities
from rag_core.documents.contextualizer import NoOpContextualizer
from rag_core.events import EventBuffer
import rag_core.search.providers.model_provider_diagnostics as diagnostics_module
from rag_core.search.lexical_sidecar import PortableLexicalSidecar
from rag_core.search.providers.embedding_cache import InMemoryChunkContextCache
from rag_core.search.providers.memory_store import InMemoryVectorStore
from rag_core.search.providers.qdrant_store import QdrantVectorStore
from rag_core.search.providers.turbopuffer_store import TurboPufferVectorStore

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
    core = RAGCore(
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
    core = RAGCore(
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
    assert voyage["support_level"] == "first_party_optional"
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
    assert sparse["providers"]["fakesparseembedder"]["support_level"] == "injected"
    event_sink = providers["event_sink"]
    assert isinstance(event_sink, dict)
    assert event_sink["configured"] == "none"
    assert "secret" not in repr(payload)


def test_describe_runtime_reports_injected_provider_category_state() -> None:
    sidecar = PortableLexicalSidecar([])
    events = EventBuffer()
    contextualizer = NoOpContextualizer()
    chunk_context_cache = InMemoryChunkContextCache()
    core = RAGCore(
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
    assert event_sink["configured"] == "buffer"
    assert contextualizer_payload["configured"] == "noop"
    assert chunk_context_cache_payload["configured"] == "in_memory"


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
    finally:
        asyncio.run(store.close())

    payload = _planned_runtime_payload(
        RAGCoreConfig(
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
    assert qdrant["query_plan_scope"] == "adapter_maximum"
    assert qdrant["query_plan"] == expected


def test_doctor_turbopuffer_query_plan_diagnostics_match_adapter_capabilities() -> None:
    store = TurboPufferVectorStore(
        namespace="docs",
        dense_dimensions=4,
        namespace_client=object(),
    )
    expected = describe_query_plan_capabilities(store.capabilities.query_plan)

    payload = _planned_runtime_payload(
        RAGCoreConfig(
            vector_store=VectorStoreConfig(
                provider="turbopuffer",
                turbopuffer=TurboPufferVectorStoreConfig(namespace="docs"),
            ),
            embedding=EmbeddingConfig(model="text-embedding-3-small", dimensions=4),
        )
    )

    vector_store = payload["vector_store"]
    assert isinstance(vector_store, dict)
    providers = vector_store["providers"]
    assert isinstance(providers, dict)
    turbopuffer = providers["turbopuffer"]
    assert isinstance(turbopuffer, dict)
    assert turbopuffer["query_plan_scope"] == "adapter_maximum"
    assert turbopuffer["query_plan"] == expected
