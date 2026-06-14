from __future__ import annotations

import asyncio

import pytest

from rag_core.search.pipeline import HybridRetrieve, PipelineContext, PipelineQuery
from rag_core.search.providers.memory_store import InMemoryVectorStore
from rag_core.search.providers.qdrant_store import QdrantVectorStore
from rag_core.search.query_plan import (
    DenseChannel,
    Prefetch,
    QueryPlan,
    SparseChannel,
    UnsupportedQueryStage,
)
from rag_core.search.provider_protocols import VectorStore

from tests.support import FakeEmbeddingProvider, FakeSparseEmbedder


def test_qdrant_specific_shape_preflight_runs_before_embedding() -> None:
    store = QdrantVectorStore(
        url=None,
        api_key=None,
        collection_name="docs",
        location=":memory:",
        dense_dimensions=4,
    )
    plan = QueryPlan(
        prefetches=(
            Prefetch(
                channel=SparseChannel(vector_field="bm25", using_query_vector="bm25"),
                limit=20,
                nested=(Prefetch(channel=DenseChannel(), limit=80),),
            ),
        ),
        final_limit=5,
    )

    try:
        _assert_rejects_before_embedding(store, plan, match="Fuse or MMR rerank")
    finally:
        asyncio.run(store.close())


def test_memory_specific_shape_preflight_runs_before_embedding() -> None:
    store = InMemoryVectorStore()
    plan = QueryPlan(
        prefetches=(
            Prefetch(
                channel=SparseChannel(
                    vector_field="unsupported_sparse",
                    using_query_vector="unsupported_sparse",
                ),
                limit=20,
            ),
        ),
        final_limit=5,
    )

    _assert_rejects_before_embedding(store, plan, match="unsupported_sparse")


def _assert_rejects_before_embedding(
    store: VectorStore,
    plan: QueryPlan,
    *,
    match: str,
) -> None:
    embedding = FakeEmbeddingProvider()
    sparse = FakeSparseEmbedder()

    async def _run() -> None:
        with pytest.raises(UnsupportedQueryStage, match=match):
            await HybridRetrieve().retrieve(
                PipelineQuery(
                    query="billing",
                    namespace="acme",
                    corpus_ids=["docs"],
                    query_plan=plan,
                ),
                PipelineContext(
                    embedding_provider=embedding,
                    sparse_embedder=sparse,
                    vector_store=store,
                ),
            )

    asyncio.run(_run())
    assert embedding.embed_query_calls == []
    assert sparse.embed_query_multi_calls == []
