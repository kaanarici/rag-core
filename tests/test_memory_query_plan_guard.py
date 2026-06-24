from __future__ import annotations

import asyncio

import pytest

from rag_core.search.planning import query_plan_preset
from rag_core.search.providers.memory_store import InMemoryVectorStore
from rag_core.search.query_plan import UnsupportedQueryStage
from rag_core.search.request_models import SearchQuery
from rag_core.search.vector_models import (
    SparseVector,
    VectorPoint,
)


def _point(
    point_id: str,
    *,
    dense: list[float],
    sparse_indices: list[int],
    text: str,
) -> VectorPoint:
    return VectorPoint(
        id=point_id,
        dense_vector=dense,
        sparse_vector=SparseVector(indices=sparse_indices, values=[1.0]),
        payload={
            "namespace": "team-space",
            "collection": "corpus-a",
            "document_id": f"doc-{point_id}",
            "content_type": "document",
            "source_type": "file",
            "text": text,
            "chunk_index": 0,
        },
    )


def _query(plan_name: str) -> SearchQuery:
    return SearchQuery(
        dense_vector=[1.0, 0.0, 0.0],
        sparse_vector=SparseVector(indices=[99], values=[1.0]),
        namespace="team-space",
        collections=["corpus-a"],
        limit=2,
        query_plan=query_plan_preset(plan_name, limit=2),
    )


async def _seeded_store() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    await store.upsert(
        [
            _point("dense", dense=[1.0, 0.0, 0.0], sparse_indices=[1], text="dense"),
            _point("sparse", dense=[0.0, 1.0, 0.0], sparse_indices=[99], text="sparse"),
        ]
    )
    return store


def test_memory_store_honors_dense_only_query_plan() -> None:
    async def _run() -> None:
        store = await _seeded_store()
        results = await store.search(_query("dense_only"))

        assert [hit.id for hit in results] == ["dense", "sparse"]
        assert results[0].score > results[1].score

    asyncio.run(_run())


def test_memory_store_honors_sparse_only_query_plan() -> None:
    async def _run() -> None:
        store = await _seeded_store()
        results = await store.search(_query("sparse_only"))

        assert [hit.id for hit in results] == ["sparse"]

    asyncio.run(_run())


def test_memory_store_honors_hybrid_rrf_query_plan() -> None:
    async def _run() -> None:
        store = await _seeded_store()
        results = await store.search(_query("hybrid_rrf"))

        assert {hit.id for hit in results} == {"dense", "sparse"}

    asyncio.run(_run())


@pytest.mark.parametrize("plan_name", ["hybrid_dbsf", "hybrid_with_mmr"])
def test_memory_store_rejects_unsupported_query_plan_stages(plan_name: str) -> None:
    async def _run() -> None:
        store = await _seeded_store()

        with pytest.raises(UnsupportedQueryStage, match="InMemoryVectorStore"):
            await store.search(_query(plan_name))

    asyncio.run(_run())


def test_memory_store_rejects_unsupported_query_plan_before_candidate_matching() -> (
    None
):
    async def _run() -> None:
        store = InMemoryVectorStore()

        with pytest.raises(UnsupportedQueryStage, match="InMemoryVectorStore"):
            await store.search(_query("hybrid_dbsf"))

    asyncio.run(_run())


@pytest.mark.parametrize("plan_name", ["dense_only", "hybrid_rrf"])
def test_memory_store_rejects_empty_dense_vector_for_dense_query_plan(
    plan_name: str,
) -> None:
    async def _run() -> None:
        store = InMemoryVectorStore()
        query = SearchQuery(
            dense_vector=[],
            sparse_vector=SparseVector(indices=[99], values=[1.0]),
            namespace="team-space",
            collections=["corpus-a"],
            limit=2,
            query_plan=query_plan_preset(plan_name, limit=2),
        )

        with pytest.raises(
            ValueError,
            match="dense query vector is required for dense query plans",
        ):
            await store.search(query)

    asyncio.run(_run())
