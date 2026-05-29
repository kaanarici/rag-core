"""End-to-end coverage that ``VectorStorePolicy`` is injectable.

Validates that a custom ``VectorStorePolicy`` flows from ``RAGCoreConfig``
through ``build_core_components`` into the indexer, the in-memory store, and
``resolve_document_id`` — so adopters can rename payload fields, the point-id
factory, and the document-id factory without forking the engine.
"""

from __future__ import annotations

import asyncio

from rag_core import RAGCore
from rag_core._engine.core_lifecycle import resolve_document_id
from rag_core.core_models import RAGCoreConfig
from rag_core.search.indexer_points import make_point_id
from rag_core.search.policy import VectorStorePolicy
from rag_core.search.providers.memory_store import InMemoryVectorStore
from rag_core.search.types import SearchQuery, SparseVector

import pytest

from tests.support import (
    FakeEmbeddingProvider,
    FakeSparseEmbedder,
    make_test_config,
)

pytestmark = [pytest.mark.integration]


def _custom_policy() -> VectorStorePolicy:
    return VectorStorePolicy(
        namespace_field="ns",
        corpus_id_field="cid",
        document_id_field="did",
        document_key_field="dkey",
        content_sha256_field="csha",
        processing_version_field="pver",
        content_type_field="ct",
        source_type_field="st",
        chunk_index_field="cidx",
        text_field="txt",
        title_field="ttl",
        tenant_payload_field="ns",
        point_id_format=lambda ns, corpus, doc, idx: f"{ns}/{corpus}/{doc}/{idx}",
        document_id_format=lambda ns, corpus, key: f"custom_{ns}_{corpus}_{key}",
    )


def _config_with_policy(policy: VectorStorePolicy) -> RAGCoreConfig:
    base = make_test_config(embedding_model="text-embedding-3-small", embedding_dimensions=4)
    return type(base)(
        qdrant=base.qdrant,
        embedding=base.embedding,
        reranker=base.reranker,
        chunking=base.chunking,
        ingest=base.ingest,
        policy=policy,
    )


def test_resolve_document_id_uses_custom_policy() -> None:
    document_id = resolve_document_id(
        namespace="team-space",
        corpus_id="corpus-1",
        document_key="/docs/guide.txt",
        document_id=None,
        policy=_custom_policy(),
    )
    assert document_id == "custom_team-space_corpus-1_/docs/guide.txt"


def test_make_point_id_uses_custom_policy() -> None:
    point_id = make_point_id(
        namespace="team-space",
        corpus_id="corpus-1",
        document_id="doc-1",
        chunk_index=0,
        policy=_custom_policy(),
    )
    assert point_id == "team-space/corpus-1/doc-1/0"


def test_policy_renames_propagate_through_ingest_and_search() -> None:
    """A custom policy must be honored end-to-end: ingest writes renamed fields,
    search reads via renamed fields, and projection back to SearchResult does
    not leak the renamed keys into ``metadata``."""

    async def _run() -> None:
        policy = _custom_policy()
        store = InMemoryVectorStore(policy=policy)
        core = RAGCore(
            _config_with_policy(policy),
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=store,
        )

        try:
            document = await core.ingest_bytes(
                file_bytes=b"alpha fox query",
                filename="guide.txt",
                mime_type="text/plain",
                namespace="team-space",
                corpus_id="corpus-1",
                path="/docs/guide.txt",
            )
            results = await store.search(
                SearchQuery(
                    dense_vector=[0.1, 0.2, 0.3, 0.4],
                    sparse_vector=SparseVector(indices=[1], values=[1.0]),
                    namespace="team-space",
                    corpus_ids=["corpus-1"],
                )
            )
        finally:
            await core.close()

        assert document.document_id == "custom_team-space_corpus-1_/docs/guide.txt"

        assert results, "expected hits via policy-renamed fields"
        first = results[0]
        assert first.corpus_id == "corpus-1"
        assert first.document_id == document.document_id
        for renamed_key in ("ns", "cid", "did", "txt", "ttl", "ct", "st", "cidx"):
            assert renamed_key not in first.metadata

    asyncio.run(_run())
