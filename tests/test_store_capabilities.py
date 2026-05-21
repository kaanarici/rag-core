"""Capability surface and registry coverage for ``VectorStore`` adapters.

What this file owns:
- Built-in adapters expose the right ``StoreCapabilities`` shape.
- ``RAGCore`` rejects a vector store missing required capabilities.
- ``QdrantIndexer`` and ``CoreIngestor`` honor missing capabilities by
  failing unsafe replacements and skipping document-record lookups when needed.
- ``VECTOR_STORES`` registry round-trip, normalization, and validation.
"""

from __future__ import annotations

import asyncio
from typing import Sequence

import pytest

from rag_core import RAGCore
from rag_core.core_ingest import CoreIngestor
from rag_core.core_models import PreparedDocument, ProcessingFingerprint
from rag_core.search.indexer import IndexRequest, QdrantIndexer
from rag_core.search.providers.memory_store import InMemoryVectorStore
from rag_core.search.providers.qdrant_store import QdrantVectorStore
from rag_core.search.providers.registry import VECTOR_STORES
from rag_core.search.types import (
    DeleteFilter,
    MetadataFilterCapabilities,
    QueryPlanCapabilities,
    SearchQuery,
    SearchResult,
    StoreCapabilities,
    StoredDocumentRecord,
    VectorPoint,
    VectorStore,
)

from tests.support import (
    FakeEmbeddingProvider,
    FakeSparseEmbedder,
    make_test_config,
)


class _MinimalVectorStore:
    """Adapter that declares no optional capabilities."""

    capabilities: StoreCapabilities = StoreCapabilities(
        per_point_delete=False,
        document_record_lookup=False,
    )

    def __init__(self) -> None:
        self.operations: list[str] = []
        self.upsert_calls: list[list[VectorPoint]] = []
        self.delete_calls: list[DeleteFilter] = []
        self.delete_point_ids_calls: list[list[str]] = []

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        self.operations.append("upsert")
        self.upsert_calls.append(list(points))

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        self.operations.append("search")
        return []

    async def delete(self, filter: DeleteFilter) -> None:
        self.operations.append("delete")
        self.delete_calls.append(filter)

    async def ensure_collection(self) -> None:
        self.operations.append("ensure_collection")

    async def check_health(self) -> dict[str, object]:
        return {"healthy": True}

    async def close(self) -> None:
        self.operations.append("close")

    async def delete_point_ids(self, point_ids: Sequence[str]) -> None:
        # Recorded but should never be invoked when per_point_delete=False.
        self.delete_point_ids_calls.append(list(point_ids))

    async def get_document_record(
        self,
        *,
        namespace: str,
        corpus_id: str,
        document_id: str | None = None,
        document_key: str | None = None,
    ) -> StoredDocumentRecord | None:
        self.operations.append("get_document_record")
        raise NotImplementedError(
            "_MinimalVectorStore does not implement get_document_record"
        )


def test_qdrant_store_declares_full_capability_surface() -> None:
    store = QdrantVectorStore(
        url=None,
        api_key=None,
        collection_name="docs",
        location=":memory:",
        dense_dimensions=4,
    )
    try:
        assert isinstance(store, VectorStore)
        assert store.capabilities == StoreCapabilities(
            per_point_delete=True,
            document_record_lookup=True,
            dense_vector_dimensions=4,
            query_plan=QueryPlanCapabilities(
                dense=True,
                sparse=True,
                hybrid_rrf=True,
                hybrid_dbsf=True,
                hybrid_weighted_rrf=True,
                mmr=True,
                boost=True,
                nested_prefetch=True,
            ),
            metadata_filter=MetadataFilterCapabilities(
                term=True,
                in_=True,
                numeric_range=True,
                string_range=False,
                geo=True,
                boolean=True,
            ),
        )
    finally:
        asyncio.run(store.close())


def test_memory_store_declares_full_capability_surface() -> None:
    store = InMemoryVectorStore()
    assert isinstance(store, VectorStore)
    assert store.capabilities == StoreCapabilities(
        per_point_delete=True,
        document_record_lookup=True,
        query_plan=QueryPlanCapabilities(dense=True, sparse=True, hybrid_rrf=True),
        metadata_filter=MetadataFilterCapabilities(
            term=True,
            in_=True,
            numeric_range=True,
            string_range=True,
            geo=True,
            boolean=True,
        ),
    )


def test_minimal_vector_store_is_vector_store_with_empty_capabilities() -> None:
    store = _MinimalVectorStore()
    assert isinstance(store, VectorStore)
    assert store.capabilities.per_point_delete is False
    assert store.capabilities.document_record_lookup is False


def test_build_core_components_rejects_store_without_record_lookup() -> None:
    config = make_test_config(embedding_dimensions=4)
    with pytest.raises(ValueError, match="document_record_lookup") as exc_info:
        RAGCore(
            config,
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=_MinimalVectorStore(),
        )
    assert "_MinimalVectorStore" in str(exc_info.value)


def test_rag_core_round_trip_against_memory_vector_store() -> None:
    """End-to-end ingest+retrieve against the in-memory store.

    Proves the ``VectorStore`` protocol is vendor-neutral end to end — nothing
    Qdrant- or TurboPuffer-specific is needed to wire ``RAGCore``.
    """

    async def _run() -> None:
        store = InMemoryVectorStore()
        core = RAGCore(
            make_test_config(
                embedding_model="text-embedding-3-small",
                embedding_dimensions=4,
            ),
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=store,
        )

        try:
            doc = await core.ingest_bytes(
                file_bytes=b"original fox query context content for retrieval",
                filename="guide.txt",
                mime_type="text/plain",
                namespace="team-space",
                corpus_id="corpus-1",
                path="/docs/guide.txt",
            )
            hits = await core.search(
                query="fox query",
                namespace="team-space",
                corpus_ids=["corpus-1"],
                limit=5,
                rerank=False,
            )
        finally:
            await core.close()

        assert doc.chunk_count >= 1
        assert hits, (
            "memory store should return at least one hit for an indexed document"
        )
        assert hits[0].document_id == doc.document_id
        assert hits[0].corpus_id == "corpus-1"

    asyncio.run(_run())


def test_indexer_rejects_unsafe_shrink_when_per_point_delete_unsupported() -> None:
    async def _run() -> None:
        store = _MinimalVectorStore()
        indexer = QdrantIndexer(
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=store,
        )

        with pytest.raises(RuntimeError, match="does not support per-point deletes"):
            await indexer.index_document(
                IndexRequest(
                    document_id="doc-1",
                    corpus_id="corpus-1",
                    namespace="team-space",
                    text="unused",
                    filename="report.txt",
                    mime_type="text/plain",
                    source_type="file",
                    existing_chunk_count=3,
                    pre_chunked_texts=["page one"],
                )
            )

        assert store.operations == []
        assert store.delete_calls == []
        assert store.delete_point_ids_calls == []
        assert store.upsert_calls == []

    asyncio.run(_run())


def test_indexer_skips_stale_handling_when_no_existing_chunk_count() -> None:
    async def _run() -> None:
        store = _MinimalVectorStore()
        indexer = QdrantIndexer(
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=store,
        )

        await indexer.index_document(
            IndexRequest(
                document_id="doc-1",
                corpus_id="corpus-1",
                namespace="team-space",
                text="unused",
                filename="report.txt",
                mime_type="text/plain",
                source_type="file",
                pre_chunked_texts=["page one"],
            )
        )

        assert store.operations == ["upsert"]

    asyncio.run(_run())


async def _noop_prepare(
    *,
    file_bytes: bytes,
    filename: str,
    mime_type: str,
    path: str | None = None,
) -> PreparedDocument:
    return PreparedDocument(
        filename=filename,
        mime_type=mime_type,
        markdown="content",
        chunks=[],
    )


def test_core_ingestor_skips_existing_lookup_when_capability_absent() -> None:
    """``document_record_lookup=False`` must not call ``get_document_record``."""

    async def _run() -> None:
        store = _MinimalVectorStore()
        indexer = QdrantIndexer(
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=store,
        )
        ingestor = CoreIngestor(
            collection_name="docs",
            source_type="file",
            embedding_model="fake-embedding",
            processing_version=ProcessingFingerprint(
                base_version="v1", source_type="file"
            ),
            store=store,
            indexer=indexer,
            sidecar=None,
            prepare_bytes=_noop_prepare,
        )

        await ingestor.ingest_bytes(
            file_bytes=b"hello",
            filename="report.txt",
            mime_type="text/plain",
            namespace="team-space",
            corpus_id="corpus-1",
        )
        assert "get_document_record" not in store.operations

    asyncio.run(_run())


def test_vector_stores_registry_creates_memory_store_by_name() -> None:
    store = VECTOR_STORES.create("memory")
    assert isinstance(store, InMemoryVectorStore)
    assert store.capabilities.document_record_lookup is True
    assert store.capabilities.per_point_delete is True


def test_vector_stores_registry_lists_known_factories() -> None:
    names = VECTOR_STORES.names()
    assert {"memory", "qdrant"}.issubset(set(names))


def test_vector_stores_registry_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="does-not-exist"):
        VECTOR_STORES.create("does-not-exist")


def test_vector_stores_registry_can_register_external_factory() -> None:
    sentinel = _MinimalVectorStore()
    VECTOR_STORES.register("test-external", lambda **_: sentinel)
    try:
        produced = VECTOR_STORES.create("test-external")
        assert produced is sentinel
        assert "test-external" in VECTOR_STORES.names()
    finally:
        VECTOR_STORES.unregister("test-external")
    assert "test-external" not in VECTOR_STORES.names()


def test_vector_stores_registry_rejects_blank_name() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        VECTOR_STORES.register("", lambda **_: _MinimalVectorStore())


def test_vector_stores_registry_normalizes_case_and_whitespace() -> None:
    sentinel = _MinimalVectorStore()
    VECTOR_STORES.register("  Whacky-Name  ", lambda **_: sentinel)
    try:
        assert "WHACKY-NAME" in VECTOR_STORES
        produced = VECTOR_STORES.create(" whacky-name ")
        assert produced is sentinel
    finally:
        VECTOR_STORES.unregister("whacky-name")
