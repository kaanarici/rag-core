from __future__ import annotations

import asyncio
from typing import cast

import pytest
from qdrant_client import models as rest

from rag_core.search.providers.qdrant_query_plan import translate_query_plan
from rag_core.search.providers.qdrant_store import QdrantVectorStore
from rag_core.search.planning import query_plan_preset
from rag_core.search.query_plan import (
    Boost,
    DenseChannel,
    Mmr,
    Prefetch,
    PrefetchFusion,
    QueryPlan,
)
from rag_core.search.query_plan import SparseChannel
from rag_core.search.query_plan import UnsupportedQueryStage
from rag_core.search.request_models import SearchQuery
from rag_core.search.vector_models import SparseVector


class _GuardedQdrantVectorStore(QdrantVectorStore):
    def __init__(self) -> None:
        super().__init__(
            url=None,
            api_key=None,
            collection_name="docs",
            location=":memory:",
            dense_dimensions=3,
        )
        self.ensure_collection_calls = 0

    async def ensure_collection(self) -> None:
        self.ensure_collection_calls += 1
        raise AssertionError("unsupported plans must fail before collection setup")


def _boost_query() -> SearchQuery:
    return SearchQuery(
        dense_vector=[1.0, 0.0, 0.0],
        sparse_vector=SparseVector(indices=[], values=[]),
        namespace="team-space",
        corpus_ids=["corpus-a"],
        limit=5,
        query_plan=QueryPlan(
            prefetches=(Prefetch(channel=DenseChannel(), limit=5),),
            boost=Boost(kind="linear_decay", field="freshness"),
            final_limit=5,
        ),
    )


def _ambiguous_nested_query() -> SearchQuery:
    return SearchQuery(
        dense_vector=[1.0, 0.0, 0.0],
        sparse_vector=SparseVector(indices=[1], values=[1.0]),
        namespace="team-space",
        corpus_ids=["corpus-a"],
        limit=5,
        query_plan=QueryPlan(
            prefetches=(
                Prefetch(
                    channel=SparseChannel(
                        vector_field="bm25",
                        using_query_vector="bm25",
                    ),
                    limit=20,
                    nested=(Prefetch(channel=DenseChannel(), limit=80),),
                ),
            ),
            final_limit=5,
        ),
    )


def _unknown_sparse_channel_query() -> SearchQuery:
    return SearchQuery(
        dense_vector=[1.0, 0.0, 0.0],
        sparse_vector=SparseVector(indices=[1], values=[1.0]),
        namespace="team-space",
        corpus_ids=["corpus-a"],
        limit=5,
        query_plan=QueryPlan(
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
        ),
    )


def _alternate_dense_channel_query() -> SearchQuery:
    return SearchQuery(
        dense_vector=[1.0, 0.0, 0.0],
        sparse_vector=SparseVector(indices=[], values=[]),
        namespace="team-space",
        corpus_ids=["corpus-a"],
        limit=5,
        query_plan=QueryPlan(
            prefetches=(Prefetch(channel=DenseChannel(vector_field="alternate"), limit=5),),
            final_limit=5,
        ),
    )


def test_qdrant_allows_boost_query_plan_past_prevalidation() -> None:
    async def _run() -> None:
        store = _GuardedQdrantVectorStore()
        try:
            with pytest.raises(
                AssertionError,
                match="unsupported plans must fail before collection setup",
            ):
                await store.search(_boost_query())

            assert store.ensure_collection_calls == 1
        finally:
            await store.close()

    asyncio.run(_run())


def test_qdrant_rejects_unknown_sparse_channel_before_collection_setup() -> None:
    async def _run() -> None:
        store = _GuardedQdrantVectorStore()
        try:
            with pytest.raises(UnsupportedQueryStage, match="unsupported_sparse"):
                await store.search(_unknown_sparse_channel_query())

            assert store.ensure_collection_calls == 0
        finally:
            await store.close()

    asyncio.run(_run())


def test_qdrant_rejects_alternate_dense_channel_before_collection_setup() -> None:
    async def _run() -> None:
        store = _GuardedQdrantVectorStore()
        try:
            with pytest.raises(UnsupportedQueryStage, match="primary dense"):
                await store.search(_alternate_dense_channel_query())

            assert store.ensure_collection_calls == 0
        finally:
            await store.close()

    asyncio.run(_run())


def test_qdrant_empty_allowlist_search_returns_empty_before_collection_setup() -> None:
    async def _run() -> None:
        store = _GuardedQdrantVectorStore()
        try:
            results = await store.search(
                SearchQuery(
                    dense_vector=[1.0, 0.0, 0.0],
                    sparse_vector=SparseVector(indices=[], values=[]),
                    namespace="team-space",
                    corpus_ids=[],
                )
            )

            assert results == []
            assert store.ensure_collection_calls == 0
        finally:
            await store.close()

    asyncio.run(_run())


def test_qdrant_sparse_only_plan_ignores_unused_dense_vector_dimensions() -> None:
    query = SearchQuery(
        dense_vector=[1.0],
        sparse_vector=SparseVector(indices=[1], values=[1.0]),
        namespace="team-space",
        corpus_ids=["corpus-a"],
        limit=5,
        query_plan=query_plan_preset("sparse_only", limit=5),
    )

    from rag_core.search.providers.qdrant_store_guards import (
        validate_qdrant_search_request,
    )

    assert validate_qdrant_search_request(query, dense_dimensions=3) == "team-space"


def test_qdrant_rejects_ambiguous_nested_prefetch_before_collection_setup() -> None:
    async def _run() -> None:
        store = _GuardedQdrantVectorStore()
        try:
            with pytest.raises(UnsupportedQueryStage, match="Fuse or MMR rerank"):
                await store.search(_ambiguous_nested_query())

            assert store.ensure_collection_calls == 0
        finally:
            await store.close()

    asyncio.run(_run())


def test_qdrant_mmr_fused_prefetch_uses_candidate_limit_before_final_limit() -> None:
    plan = QueryPlan(
        prefetches=(
            Prefetch(channel=DenseChannel(), limit=80),
            Prefetch(channel=SparseChannel(), limit=80),
        ),
        fuse=PrefetchFusion(),
        rerank=Mmr(diversity=0.5, limit=40),
        final_limit=5,
    )
    query = SearchQuery(
        dense_vector=[1.0, 0.0, 0.0],
        sparse_vector=SparseVector(indices=[1], values=[1.0]),
        namespace="team-space",
        corpus_ids=["corpus-a"],
        limit=5,
    )

    translated = translate_query_plan(
        plan,
        query=query,
        qdrant_filter=rest.Filter(),
    )

    assert len(translated.prefetch) == 1
    wrapper = translated.prefetch[0]
    assert wrapper.limit == 40
    assert translated.limit == 5
    assert isinstance(translated.query, rest.NearestQuery)
    assert translated.query.mmr is not None
    assert translated.query.mmr.candidates_limit == 40
    nested = cast(list[rest.Prefetch], wrapper.prefetch)
    assert [prefetch.limit for prefetch in nested] == [80, 80]
