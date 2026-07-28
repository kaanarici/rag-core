from __future__ import annotations

import asyncio
from typing import Any

import pytest

from rag_core import Config, Document, RAGCore
from rag_core.config import (
    DEMO_EMBEDDING_MODEL,
    DEMO_EMBEDDING_PROVIDER,
    EmbeddingConfig,
    QdrantConfig,
)
from rag_core.config.env_config import build_config_from_env
from rag_core.core import Engine
from rag_core.demo import DemoEmbeddingProvider, build_demo_core
from rag_core.search.policy import CollectionPolicy
from rag_core.search.vector_models import SearchResult

from tests.support import (
    FakeEmbeddingProvider,
    FakeSparseEmbedder,
    RecordingVectorStore,
    make_test_config,
    make_search_result,
)


def _make_rag(
    store: RecordingVectorStore,
    *,
    tenant_id: str = "acme",
    index: str = "company-docs",
    document_ids: tuple[str, ...] | None = ("doc-1",),
) -> RAGCore:
    engine = Engine(
        make_test_config(
            qdrant_collection="rag_core_facade",
            embedding_dimensions=4,
        ),
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store,
    )
    return RAGCore(
        engine,
        tenant_id=tenant_id,
        index=index,
        document_ids=document_ids,
    )


def test_rag_core_binds_scope_across_ingest_search_and_delete() -> None:
    async def scenario() -> RecordingVectorStore:
        store = RecordingVectorStore()
        rag = _make_rag(store)
        try:
            ingested = await rag.ingest(
                Document(
                    id="doc-1",
                    key="policies/billing.md",
                    content=b"Invoices can be paid by bank transfer.",
                    content_type="text/markdown",
                )
            )
            await rag.search("How can invoices be paid?", document_ids=("doc-1",))
            with pytest.raises(ValueError, match="outside the configured retrieval scope"):
                await rag.search("private", document_ids=("doc-2",))
            with pytest.raises(ValueError, match="Document.id is required"):
                await rag.ingest(
                    Document(
                        key="unscoped.md",
                        content=b"unscoped",
                        content_type="text/markdown",
                    )
                )
            deleted = await rag.delete("doc-1")
        finally:
            await rag.close()

        assert ingested.document_id == "doc-1"
        assert ingested.status == "created"
        assert deleted.namespace == "acme"
        assert deleted.collection == "company-docs"
        return store

    store = asyncio.run(scenario())

    point = store.upsert_calls[0][0]
    assert point.payload["namespace"] == "acme"
    assert point.payload["collection"] == "company-docs"
    assert store.search_calls[0].namespace == "acme"
    assert store.search_calls[0].collections == ["company-docs"]
    assert store.search_calls[0].document_ids == ["doc-1"]
    assert len(store.search_calls) == 1
    assert store.delete_calls[0].namespace == "acme"
    assert store.delete_calls[0].collection == "company-docs"
    assert store.delete_calls[0].document_id == "doc-1"


def test_rag_core_tool_reuses_bound_openai_agents_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    sentinel = object()

    def fake_builder(core: object, **kwargs: Any) -> object:
        captured["core"] = core
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        "rag_core.integrations.openai_agents.build_retrieve_context_tool",
        fake_builder,
    )
    rag = _make_rag(RecordingVectorStore())

    assert rag.tool() is sentinel
    assert captured["namespace"] == "acme"
    assert captured["collection"] == "company-docs"
    assert captured["document_ids"] == ("doc-1",)
    assert captured["default_rerank"] is False
    assert captured["default_use_lexical_search"] is False
    assert captured["expose_strategy_options"] is False
    assert not hasattr(rag, "context")
    assert not hasattr(rag, "delete_collection")

    asyncio.run(rag.close())


def test_rag_core_rejects_scope_widening_before_engine_calls() -> None:
    with pytest.raises(ValueError, match="namespace='other'"):
        RAGCore(
            Engine(
                make_test_config().__class__(
                    collection_policy=CollectionPolicy(bound_namespace="other")
                ),
                embedding_provider=FakeEmbeddingProvider(),
                sparse_embedder=FakeSparseEmbedder(),
                vector_store=RecordingVectorStore(),
            ),
            tenant_id="acme",
            index="company-docs",
        )

    with pytest.raises(ValueError, match="index"):
        RAGCore(make_test_config(), tenant_id="acme", index=" ")


def test_rag_core_runs_on_qdrant_memory() -> None:
    async def scenario() -> None:
        rag = RAGCore(
            build_demo_core(store_collection="rag_core_facade_integration"),
            tenant_id="acme",
            index="company-docs",
        )
        async with rag:
            ingested = await rag.ingest(
                Document(
                    key="billing.md",
                    content=b"Invoices can be paid by card or ACH.",
                    content_type="text/markdown",
                )
            )
            result = await rag.search("How can invoices be paid?")
            deleted = await rag.delete(ingested.document_id)

        assert result.evidence
        assert result.evidence[0].document_id == ingested.document_id
        assert (
            result.evidence[0].source_id
            == f"acme:company-docs:{ingested.document_id}#chunk-0"
        )
        assert result.evidence[0].locator.chunk_index == 0
        assert deleted.index_deleted is True

    asyncio.run(scenario())


def test_rag_core_collapses_exact_duplicate_evidence_deterministically() -> None:
    duplicate_a = make_search_result(
        id="chunk-a",
        text="Invoices can be paid by ACH.",
        score=0.8,
        namespace="acme",
        collection="company-docs",
        document_id="doc-a",
        document_key="a.md",
    )
    duplicate_b = make_search_result(
        id="chunk-b",
        text="Invoices can be paid by ACH.",
        score=0.8,
        namespace="acme",
        collection="company-docs",
        document_id="doc-b",
        document_key="b.md",
    )
    distinct = make_search_result(
        id="chunk-c",
        text="Card payments are also supported.",
        score=0.7,
        namespace="acme",
        collection="company-docs",
        document_id="doc-c",
        document_key="c.md",
    )

    async def search(
        results: list[SearchResult],
    ) -> tuple[tuple[str | None, ...], tuple[str | None, ...], int]:
        store = RecordingVectorStore(search_results=results)
        rag = _make_rag(store, document_ids=None)
        try:
            retrieved = await rag.search("How can invoices be paid?", limit=2)
        finally:
            await rag.close()
        primary_ids = tuple(item.document_id for item in retrieved.evidence)
        equivalent_ids = tuple(
            item.document_id for item in retrieved.evidence[0].equivalent_sources
        )
        assert all(
            not item.equivalent_sources
            for item in retrieved.evidence[0].equivalent_sources
        )
        return primary_ids, equivalent_ids, store.search_calls[0].limit

    first = asyncio.run(search([duplicate_b, duplicate_a, distinct]))
    second = asyncio.run(search([duplicate_a, duplicate_b, distinct]))

    assert first == second
    assert first == (("doc-a", "doc-c"), ("doc-b",), 20)


def test_rag_core_duplicate_collapsing_is_exact_and_type_aware() -> None:
    results = [
        make_search_result(id="upper", text="Alpha", result_type=None),
        make_search_result(id="lower", text="alpha", result_type=None),
        make_search_result(id="space", text="Alpha ", result_type=None),
        make_search_result(id="image", text="Alpha", result_type="image"),
    ]

    async def scenario() -> None:
        rag = _make_rag(
            RecordingVectorStore(search_results=results),
            document_ids=None,
        )
        try:
            retrieved = await rag.search("alpha", limit=4)
        finally:
            await rag.close()
        assert [item.text for item in retrieved.evidence] == [
            "Alpha",
            "alpha",
            "Alpha ",
            "Alpha",
        ]
        assert all(not item.equivalent_sources for item in retrieved.evidence)

    asyncio.run(scenario())


def test_rag_core_ingest_is_idempotent_and_collapses_duplicate_sources_on_qdrant() -> None:
    class CountingEmbeddingProvider(DemoEmbeddingProvider):
        def __init__(self) -> None:
            super().__init__(dimensions=64)
            self.embedded_text_count = 0

        async def embed_texts(self, texts: list[str]) -> list[list[float]]:
            self.embedded_text_count += len(texts)
            return await super().embed_texts(texts)

    async def scenario() -> None:
        embedding = CountingEmbeddingProvider()
        engine = Engine(
            Config(
                qdrant=QdrantConfig(
                    location=":memory:",
                    store_collection="rag_core_idempotency_integration",
                    dimension_aware_collection=False,
                ),
                embedding=EmbeddingConfig(
                    provider=DEMO_EMBEDDING_PROVIDER,
                    model=DEMO_EMBEDDING_MODEL,
                    dimensions=64,
                ),
            ),
            embedding_provider=embedding,
        )
        rag = RAGCore(engine, tenant_id="acme", index="company-docs")
        content = b"Invoices can be paid by ACH."
        async with rag:
            first = await rag.ingest(
                Document(
                    id="doc-a",
                    key="a.md",
                    content=content,
                    content_type="text/markdown",
                )
            )
            embedded_after_first = embedding.embedded_text_count
            unchanged = await rag.ingest(
                Document(
                    id="doc-a",
                    key="a.md",
                    content=content,
                    content_type="text/markdown",
                )
            )
            health_after_unchanged = await engine._store.check_health()
            second_source = await rag.ingest(
                Document(
                    id="doc-b",
                    key="b.md",
                    content=content,
                    content_type="text/markdown",
                )
            )
            retrieved = await rag.search("How can invoices be paid?", limit=2)

        assert first.status == "created"
        assert unchanged.status == "unchanged"
        assert unchanged.document_id == first.document_id
        assert unchanged.content_hash == first.content_hash
        assert embedding.embedded_text_count == embedded_after_first + 1
        assert health_after_unchanged["points_count"] == 1
        assert second_source.status == "created"
        assert len(retrieved.evidence) == 1
        source_ids = {
            retrieved.evidence[0].document_id,
            *(
                source.document_id
                for source in retrieved.evidence[0].equivalent_sources
            ),
        }
        assert source_ids == {"doc-a", "doc-b"}

    asyncio.run(scenario())


def test_rag_core_reindexes_payload_metadata_changes_on_qdrant() -> None:
    class CountingEmbeddingProvider(DemoEmbeddingProvider):
        def __init__(self) -> None:
            super().__init__(dimensions=64)
            self.embedded_text_count = 0

        async def embed_texts(self, texts: list[str]) -> list[list[float]]:
            self.embedded_text_count += len(texts)
            return await super().embed_texts(texts)

    async def scenario() -> None:
        embedding = CountingEmbeddingProvider()
        rag = RAGCore(
            Engine(
                Config(
                    qdrant=QdrantConfig(
                        location=":memory:",
                        store_collection="rag_core_metadata_update_integration",
                        dimension_aware_collection=False,
                    ),
                    embedding=EmbeddingConfig(
                        provider=DEMO_EMBEDDING_PROVIDER,
                        model=DEMO_EMBEDDING_MODEL,
                        dimensions=64,
                    ),
                ),
                embedding_provider=embedding,
            ),
            tenant_id="acme",
            index="company-docs",
        )
        document = Document(
            id="doc-a",
            key="billing.md",
            content=b"Invoices can be paid by ACH.",
            content_type="text/markdown",
            metadata={"title": "Original billing guide"},
        )
        async with rag:
            created = await rag.ingest(document)
            embedded_after_create = embedding.embedded_text_count
            unchanged = await rag.ingest(document)
            embedded_after_unchanged = embedding.embedded_text_count
            replaced = await rag.ingest(
                Document(
                    id=document.id,
                    key=document.key,
                    content=document.content,
                    content_type=document.content_type,
                    metadata={"title": "Updated billing guide"},
                )
            )
            embedded_after_replace = embedding.embedded_text_count
            retrieved = await rag.search("How can invoices be paid?")

        assert created.status == "created"
        assert unchanged.status == "unchanged"
        assert embedded_after_unchanged == embedded_after_create
        assert replaced.status == "replaced"
        assert embedded_after_replace == embedded_after_create + 1
        assert retrieved.evidence[0].title == "Updated billing guide"

    asyncio.run(scenario())


def test_rag_core_tool_uses_the_same_duplicate_collapsing_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_function_tool(**kwargs: object) -> object:
        def decorate(function: object) -> object:
            return function

        return decorate

    monkeypatch.setattr(
        "rag_core.integrations.openai_agents.import_agents_function_tool",
        lambda: fake_function_tool,
    )

    async def scenario() -> None:
        rag = RAGCore(
            build_demo_core(store_collection="rag_core_tool_dedup_integration"),
            tenant_id="acme",
            index="company-docs",
        )
        content = b"Invoices can be paid by ACH."
        async with rag:
            for document_id in ("doc-a", "doc-b"):
                await rag.ingest(
                    Document(
                        id=document_id,
                        key=f"{document_id}.md",
                        content=content,
                        content_type="text/markdown",
                    )
                )
            retrieved = await rag.search("How can invoices be paid?", limit=2)
            tool = rag.tool()
            tool_result = await tool(
                query="How can invoices be paid?",
                limit=2,
            )

        assert len(retrieved.evidence) == 1
        assert len(retrieved.evidence[0].equivalent_sources) == 1
        assert isinstance(tool_result, dict)
        snippets = tool_result["snippets"]
        assert isinstance(snippets, list)
        assert len(snippets) == 1

    asyncio.run(scenario())


def test_rag_core_scope_does_not_rename_an_app_owned_physical_collection() -> None:
    rag = RAGCore(
        Config(
            qdrant=QdrantConfig(
                location=":memory:",
                store_collection="app_owned_index",
                dimension_aware_collection=False,
            ),
            embedding=EmbeddingConfig(
                provider=DEMO_EMBEDDING_PROVIDER,
                model=DEMO_EMBEDDING_MODEL,
                dimensions=64,
            ),
        ),
        tenant_id="acme",
        index="company-docs",
    )
    try:
        assert rag._engine._collection_name == "app_owned_index"
        assert rag._engine._config.collection_policy is not None
        assert rag._engine._config.collection_policy.bound_namespace == "acme"
        assert rag._engine._config.collection_policy.allowed_collections is None
    finally:
        asyncio.run(rag.close())


def test_from_env_builds_scoped_dense_qdrant_facade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_CORE_TENANT_ID", "acme")
    monkeypatch.setenv("RAG_CORE_QDRANT_LOCATION", ":memory:")
    monkeypatch.delenv("RAG_CORE_QDRANT_URL", raising=False)
    monkeypatch.setenv("RAG_CORE_EMBEDDING_PROVIDER", DEMO_EMBEDDING_PROVIDER)
    monkeypatch.setenv("RAG_CORE_EMBEDDING_MODEL", DEMO_EMBEDDING_MODEL)
    monkeypatch.setenv("RAG_CORE_EMBEDDING_DIMENSIONS", "64")

    rag = RAGCore.from_env(index="company-docs")
    try:
        assert rag.tenant_id == "acme"
        assert rag.index == "company-docs"
        assert rag._engine._sparse is None
        assert rag._engine._store.capabilities.query_plan.dense is True
        assert rag._engine._store.capabilities.query_plan.sparse is False
    finally:
        asyncio.run(rag.close())


def test_from_env_fails_closed_without_tenant_or_store_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAG_CORE_TENANT_ID", raising=False)
    with pytest.raises(ValueError, match="RAG_CORE_TENANT_ID"):
        RAGCore.from_env(index="company-docs")

    monkeypatch.setenv("RAG_CORE_TENANT_ID", "acme")
    monkeypatch.delenv("RAG_CORE_QDRANT_URL", raising=False)
    monkeypatch.delenv("RAG_CORE_QDRANT_LOCATION", raising=False)
    with pytest.raises(ValueError, match="requires exactly one"):
        RAGCore.from_env(index="company-docs")


def test_from_env_maps_existing_pgvector_and_turbopuffer_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_CORE_VECTOR_STORE", "pgvector")
    monkeypatch.setenv("RAG_CORE_PGVECTOR_DSN", "postgresql://db/rag")
    pgvector = build_config_from_env()
    assert pgvector.vector_store.provider == "pgvector"
    assert pgvector.vector_store.pgvector.dsn == "postgresql://db/rag"

    monkeypatch.setenv("RAG_CORE_VECTOR_STORE", "turbopuffer")
    monkeypatch.setenv("RAG_CORE_TURBOPUFFER_NAMESPACE", "company-docs")
    turbopuffer = build_config_from_env()
    assert turbopuffer.vector_store.provider == "turbopuffer"
    assert turbopuffer.vector_store.turbopuffer.namespace == "company-docs"


def test_default_config_does_not_construct_sparse_machinery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_sparse_init(*args: object, **kwargs: object) -> object:
        raise AssertionError("dense configuration constructed sparse machinery")

    monkeypatch.setattr(
        "rag_core.search.providers.sparse.FastEmbedSparseEmbedder",
        unexpected_sparse_init,
    )
    rag = RAGCore(
        Config(
            qdrant=QdrantConfig(location=":memory:"),
            embedding=EmbeddingConfig(
                provider=DEMO_EMBEDDING_PROVIDER,
                model=DEMO_EMBEDDING_MODEL,
                dimensions=64,
            ),
        ),
        tenant_id="acme",
        index="company-docs",
    )
    try:
        assert rag._engine._sparse is None
    finally:
        asyncio.run(rag.close())
