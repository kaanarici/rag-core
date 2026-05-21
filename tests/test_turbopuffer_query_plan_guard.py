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
        return SimpleNamespace(rows=[])


def _query(*, plan_name: str | None = None, limit: int = 5) -> SearchQuery:
    return SearchQuery(
        dense_vector=[1.0, 0.0, 0.0],
        sparse_vector=SparseVector(indices=[], values=[]),
        namespace="team-space",
        corpus_ids=["corpus-a"],
        limit=limit,
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


@pytest.mark.parametrize("plan_name", ["sparse_only", "hybrid_rrf", "hybrid_dbsf"])
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
