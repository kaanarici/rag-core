from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from rag_core.search.planning import query_plan_preset
from rag_core.search.providers.turbopuffer_payloads import TURBOPUFFER_BM25_TEXT_FIELD
from rag_core.search.providers.turbopuffer_store import TurboPufferVectorStore
from rag_core.search.query_plan import (
    DenseChannel,
    Prefetch,
    PrefetchFusion,
    QueryPlan,
    SparseChannel,
    UnsupportedQueryStage,
)
from rag_core.search.request_models import SearchQuery
from rag_core.search.sparse_channels import SECONDARY_SPARSE_CHANNEL
from rag_core.search.vector_models import SparseVector


class _RecordingNamespace:
    def __init__(self) -> None:
        self.query_calls: list[dict[str, object]] = []

    async def metadata(self) -> object:
        return SimpleNamespace()

    async def write(self, **kwargs: object) -> object:
        return SimpleNamespace(rows_remaining=False)

    async def query(self, **kwargs: object) -> object:
        self.query_calls.append(kwargs)
        if "queries" in kwargs:
            queries = kwargs["queries"]
            assert isinstance(queries, list)
            return SimpleNamespace(results=[SimpleNamespace(rows=[]) for _ in queries])
        return SimpleNamespace(rows=[])


def _query(
    *,
    plan_name: str | None = None,
    query_plan: QueryPlan | None = None,
    limit: int = 5,
    lexical_query: str | None = "fox jumping",
) -> SearchQuery:
    resolved_plan = query_plan
    if resolved_plan is None and plan_name is not None:
        resolved_plan = query_plan_preset(plan_name, limit=limit)
    return SearchQuery(
        dense_vector=[1.0, 0.0, 0.0],
        sparse_vector=SparseVector(indices=[1, 3], values=[0.5, 0.25]),
        namespace="team-space",
        collections=["corpus-a"],
        limit=limit,
        lexical_query=lexical_query,
        query_plan=resolved_plan,
    )


def test_turbopuffer_honors_dense_only_query_plan_limit() -> None:
    async def _run() -> None:
        namespace = _RecordingNamespace()
        store = TurboPufferVectorStore(
            namespace="docs",
            dense_dimensions=3,
            namespace_client=namespace,
        )

        await store.search(_query(plan_name="dense_only", limit=7))

        assert namespace.query_calls[0]["top_k"] == 28
        assert namespace.query_calls[0]["rank_by"] == ("vector", "ANN", [1.0, 0.0, 0.0])

    asyncio.run(_run())


def test_turbopuffer_honors_bm25_query_plan_limit() -> None:
    async def _run() -> None:
        namespace = _RecordingNamespace()
        store = TurboPufferVectorStore(
            namespace="docs",
            dense_dimensions=3,
            namespace_client=namespace,
        )

        await store.search(_query(plan_name="sparse_only", limit=5))

        assert namespace.query_calls[0]["top_k"] == 20
        assert namespace.query_calls[0]["rank_by"] == (
            TURBOPUFFER_BM25_TEXT_FIELD,
            "BM25",
            "fox jumping",
        )

    asyncio.run(_run())


def test_turbopuffer_honors_hybrid_rrf_query_plan_limits() -> None:
    async def _run() -> None:
        namespace = _RecordingNamespace()
        store = TurboPufferVectorStore(
            namespace="docs",
            dense_dimensions=3,
            namespace_client=namespace,
        )

        await store.search(_query(plan_name="hybrid_rrf", limit=5))

        rank_by_values = [call["rank_by"] for call in namespace.query_calls]
        assert ("vector", "ANN", [1.0, 0.0, 0.0]) in rank_by_values
        assert (TURBOPUFFER_BM25_TEXT_FIELD, "BM25", "fox jumping") in rank_by_values
        assert [call["top_k"] for call in namespace.query_calls] == [20, 20]

    asyncio.run(_run())


def test_turbopuffer_rejects_bm25_query_plan_without_lexical_query() -> None:
    async def _run() -> None:
        namespace = _RecordingNamespace()
        store = TurboPufferVectorStore(
            namespace="docs",
            dense_dimensions=3,
            namespace_client=namespace,
        )

        with pytest.raises(
            UnsupportedQueryStage, match="requires SearchQuery.lexical_query"
        ):
            await store.search(
                _query(plan_name="sparse_only", limit=5, lexical_query=None)
            )
        assert namespace.query_calls == []

    asyncio.run(_run())


def test_turbopuffer_rejects_non_primary_bm25_channel() -> None:
    async def _run() -> None:
        namespace = _RecordingNamespace()
        store = TurboPufferVectorStore(
            namespace="docs",
            dense_dimensions=3,
            namespace_client=namespace,
        )
        plan = QueryPlan(
            prefetches=(
                Prefetch(
                    channel=SparseChannel(
                        vector_field=SECONDARY_SPARSE_CHANNEL,
                        using_query_vector=SECONDARY_SPARSE_CHANNEL,
                    ),
                    limit=20,
                ),
            ),
            final_limit=5,
        )

        with pytest.raises(UnsupportedQueryStage, match="primary BM25 channel"):
            await store.search(_query(query_plan=plan, lexical_query="fox"))

        assert namespace.query_calls == []

    asyncio.run(_run())


def test_turbopuffer_rejects_non_positive_hybrid_rrf_k() -> None:
    async def _run() -> None:
        namespace = _RecordingNamespace()
        store = TurboPufferVectorStore(
            namespace="docs",
            dense_dimensions=3,
            namespace_client=namespace,
        )
        plan = QueryPlan(
            prefetches=(
                Prefetch(channel=DenseChannel(), limit=20),
                Prefetch(channel=SparseChannel(), limit=20),
            ),
            fuse=PrefetchFusion(kind="rrf", rrf_k=0),
            final_limit=5,
        )

        with pytest.raises(UnsupportedQueryStage, match="positive rrf_k"):
            await store.search(_query(query_plan=plan))

        assert namespace.query_calls == []

    asyncio.run(_run())


@pytest.mark.parametrize("plan_name", ["hybrid_dbsf", "hybrid_with_mmr"])
def test_turbopuffer_rejects_unsupported_query_plans(plan_name: str) -> None:
    async def _run() -> None:
        namespace = _RecordingNamespace()
        store = TurboPufferVectorStore(
            namespace="docs",
            dense_dimensions=3,
            namespace_client=namespace,
        )

        with pytest.raises(UnsupportedQueryStage, match="turbopuffer adapter"):
            await store.search(_query(plan_name=plan_name))

        assert namespace.query_calls == []

    asyncio.run(_run())
