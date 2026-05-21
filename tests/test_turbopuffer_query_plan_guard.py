from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from rag_core.search.planning import query_plan_preset
from rag_core.search.providers.turbopuffer_store import TurboPufferVectorStore
from rag_core.search.query_plan import UnsupportedQueryStage
from rag_core.search.types import SearchQuery, SparseVector


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
            return SimpleNamespace(
                results=[SimpleNamespace(rows=[]) for _ in queries]
            )
        return SimpleNamespace(rows=[])


def _query(
    *,
    plan_name: str | None = None,
    limit: int = 5,
    lexical_query: str | None = "fox jumping",
) -> SearchQuery:
    return SearchQuery(
        dense_vector=[1.0, 0.0, 0.0],
        sparse_vector=SparseVector(indices=[1, 3], values=[0.5, 0.25]),
        namespace="team-space",
        corpus_ids=["corpus-a"],
        limit=limit,
        lexical_query=lexical_query,
        query_plan=(
            query_plan_preset(plan_name, limit=limit) if plan_name is not None else None
        ),
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

        assert namespace.query_calls[0]["top_k"] == 7
        assert namespace.query_calls[0]["rank_by"] == ("vector", "ANN", [1.0, 0.0, 0.0])

    asyncio.run(_run())


def test_turbopuffer_hybrid_rrf_uses_multi_query_with_bm25() -> None:
    async def _run() -> None:
        namespace = _RecordingNamespace()
        store = TurboPufferVectorStore(
            namespace="docs",
            dense_dimensions=3,
            namespace_client=namespace,
        )

        await store.search(_query(plan_name="hybrid_rrf", limit=5))

        assert len(namespace.query_calls) == 1
        subqueries = namespace.query_calls[0]["queries"]
        assert isinstance(subqueries, list)
        assert len(subqueries) == 2
        assert subqueries[0]["rank_by"] == ("vector", "ANN", [1.0, 0.0, 0.0])
        assert subqueries[1]["rank_by"] == ("text", "BM25", "fox jumping")

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
