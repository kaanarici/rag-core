"""Tests for the Qdrant translator that turns a QueryPlan into a query_points call."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest
from qdrant_client import models as rest

from rag_core.search.providers.qdrant_query_plan import translate_query_plan
from rag_core.search.providers.qdrant_search import (
    _default_query_plan_for_available_sparse_channels,
)
from rag_core.search.providers.qdrant_store import QdrantVectorStore
from rag_core.search.query_plan import (
    Boost,
    DEFAULT_RRF_K,
    DenseChannel,
    Mmr,
    Prefetch,
    PrefetchFusion,
    QueryPlan,
    SparseChannel,
    UnsupportedQueryStage,
)
from rag_core.search.request_models import SearchQuery
from rag_core.search.vector_models import SparseVector


def _make_query(**overrides: Any) -> SearchQuery:
    base: dict[str, Any] = dict(
        dense_vector=[0.1, 0.2, 0.3],
        sparse_vector=SparseVector(indices=[1, 2], values=[1.0, 2.0]),
        namespace="team-space",
        corpus_ids=["corpus-a"],
        sparse_vectors={"splade": SparseVector(indices=[3], values=[3.0])},
        limit=20,
    )
    base.update(overrides)
    return SearchQuery(**base)


def _empty_filter() -> rest.Filter:
    return rest.Filter()


class _FakeQdrantClient:
    def __init__(
        self,
        *,
        existing_names: list[str],
        sparse_names: list[str] | None = None,
    ) -> None:
        self._existing_names = existing_names
        self._sparse_names = sparse_names
        self.query_points_calls: list[dict[str, Any]] = []

    async def get_collections(self) -> object:
        return SimpleNamespace(
            collections=[SimpleNamespace(name=name) for name in self._existing_names]
        )

    async def get_collection(self, *, collection_name: str) -> object:
        return _FakeCollectionInfo(size=3, sparse_names=self._sparse_names)

    async def query_points(self, **kwargs: Any) -> object:
        self.query_points_calls.append(kwargs)
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id=str(uuid4()),
                    score=0.9,
                    payload={
                        "text": "x",
                        "content_type": "document",
                        "source_type": "file",
                    },
                )
            ]
        )

    async def close(self) -> None:
        pass


class _FakeNamedSparse:
    def __init__(self, names: list[str]) -> None:
        self._NamedSparseVectorStruct__root = {name: object() for name in names}


class _FakeNamedDense:
    def __init__(self, size: int) -> None:
        self._NamedVectorStruct__root = {"": type("Dense", (), {"size": size})()}


class _FakeConfigParams:
    def __init__(self, *, size: int, sparse_names: list[str] | None) -> None:
        self.vectors = _FakeNamedDense(size=size)
        self.sparse_vectors = (
            _FakeNamedSparse(sparse_names) if sparse_names is not None else None
        )


class _FakeCollectionInfo:
    def __init__(self, *, size: int, sparse_names: list[str] | None = None) -> None:
        self.config = type(
            "Config",
            (),
            {"params": _FakeConfigParams(size=size, sparse_names=sparse_names)},
        )()


def _store_with_fake_client(
    client: object,
    *,
    sparse_names: frozenset[str] | None = None,
) -> QdrantVectorStore:
    """Build a QdrantVectorStore and swap its client for a fake.

    The store creates its own AsyncQdrantClient in __init__; we substitute the
    fake afterward and pre-flip the collection-ready flag so the fake doesn't
    need to implement the bootstrap path.
    """
    store = QdrantVectorStore(
        url=None,
        api_key=None,
        collection_name="docs",
        location=":memory:",
        dense_dimensions=3,
        quantization_enabled=False,
    )
    store._client = cast(Any, client)
    store._collection_state.ready = True
    store._collection_state.available_sparse_vector_names = sparse_names or frozenset(
        {"bm25", "splade"}
    )
    return store


def test_default_plan_translates_to_two_prefetches_and_rrf_fusion() -> None:
    plan = QueryPlan(
        prefetches=(
            Prefetch(channel=DenseChannel(), limit=80),
            Prefetch(
                channel=SparseChannel(vector_field="bm25", using_query_vector="bm25"),
                limit=80,
            ),
        ),
        fuse=PrefetchFusion(kind="rrf", rrf_k=DEFAULT_RRF_K),
        final_limit=20,
    )
    translated = translate_query_plan(
        plan,
        query=_make_query(),
        qdrant_filter=_empty_filter(),
        available_sparse_names={"bm25"},
    )

    assert len(translated.prefetch) == 2
    assert [p.using for p in translated.prefetch] == ["", "bm25"]
    assert [p.limit for p in translated.prefetch] == [80, 80]
    assert isinstance(translated.query, rest.FusionQuery)
    assert translated.query.fusion == rest.Fusion.RRF
    assert translated.limit == 20


def test_three_channel_plan_produces_three_prefetches() -> None:
    plan = QueryPlan(
        prefetches=(
            Prefetch(channel=DenseChannel(), limit=80),
            Prefetch(
                channel=SparseChannel(vector_field="bm25", using_query_vector="bm25"),
                limit=80,
            ),
            Prefetch(
                channel=SparseChannel(
                    vector_field="splade", using_query_vector="splade"
                ),
                limit=80,
            ),
        ),
        fuse=PrefetchFusion(),
        final_limit=10,
    )
    translated = translate_query_plan(
        plan,
        query=_make_query(),
        qdrant_filter=_empty_filter(),
        available_sparse_names={"bm25", "splade"},
    )
    assert [p.using for p in translated.prefetch] == ["", "bm25", "splade"]


def test_dbsf_fusion_translates_to_dbsf_fusion_query() -> None:
    plan = QueryPlan(
        prefetches=(
            Prefetch(channel=DenseChannel(), limit=80),
            Prefetch(channel=SparseChannel(), limit=80),
        ),
        fuse=PrefetchFusion(kind="dbsf"),
        final_limit=20,
    )
    translated = translate_query_plan(
        plan, query=_make_query(), qdrant_filter=_empty_filter()
    )
    assert isinstance(translated.query, rest.FusionQuery)
    assert translated.query.fusion == rest.Fusion.DBSF


def test_custom_rrf_k_translates_to_rrf_query() -> None:
    plan = QueryPlan(
        prefetches=(
            Prefetch(channel=DenseChannel(), limit=80),
            Prefetch(channel=SparseChannel(), limit=80),
        ),
        fuse=PrefetchFusion(kind="rrf", rrf_k=42),
        final_limit=20,
    )
    translated = translate_query_plan(
        plan, query=_make_query(), qdrant_filter=_empty_filter()
    )
    assert isinstance(translated.query, rest.RrfQuery)
    assert translated.query.rrf.k == 42


def test_weighted_rrf_translates_with_weights() -> None:
    plan = QueryPlan(
        prefetches=(
            Prefetch(channel=DenseChannel(), limit=80),
            Prefetch(channel=SparseChannel(), limit=80),
        ),
        fuse=PrefetchFusion(kind="weighted_rrf", weights=(2.0, 1.0)),
        final_limit=20,
    )
    translated = translate_query_plan(
        plan, query=_make_query(), qdrant_filter=_empty_filter()
    )
    assert isinstance(translated.query, rest.RrfQuery)
    assert translated.query.rrf.weights == [2.0, 1.0]


def test_single_channel_plan_preserves_query_filter() -> None:
    plan = QueryPlan(
        prefetches=(Prefetch(channel=DenseChannel(), limit=10),),
        final_limit=5,
    )
    query_filter = rest.Filter(
        must=[
            rest.FieldCondition(
                key="namespace",
                match=rest.MatchValue(value="team-space"),
            )
        ]
    )

    translated = translate_query_plan(
        plan,
        query=_make_query(),
        qdrant_filter=query_filter,
    )

    assert translated.prefetch == []
    assert translated.query_filter is query_filter
    assert translated.using == ""


def test_weighted_rrf_rejects_weight_count_mismatch() -> None:
    plan = QueryPlan(
        prefetches=(
            Prefetch(channel=DenseChannel(), limit=80),
            Prefetch(channel=SparseChannel(), limit=80),
        ),
        fuse=PrefetchFusion(kind="weighted_rrf", weights=(1.0,)),
        final_limit=20,
    )
    with pytest.raises(UnsupportedQueryStage, match="one weight per prefetch"):
        translate_query_plan(plan, query=_make_query(), qdrant_filter=_empty_filter())


def test_mmr_rerank_wraps_fused_prefetch() -> None:
    plan = QueryPlan(
        prefetches=(
            Prefetch(channel=DenseChannel(), limit=80),
            Prefetch(channel=SparseChannel(), limit=80),
        ),
        fuse=PrefetchFusion(),
        rerank=Mmr(diversity=0.5, limit=20),
        final_limit=20,
    )
    translated = translate_query_plan(
        plan, query=_make_query(), qdrant_filter=_empty_filter()
    )
    assert len(translated.prefetch) == 1
    wrapper = translated.prefetch[0]
    assert wrapper.prefetch is not None
    nested_list = cast(list[rest.Prefetch], wrapper.prefetch)
    assert len(nested_list) == 2
    assert isinstance(translated.query, rest.NearestQuery)
    assert translated.query.mmr is not None
    assert translated.query.mmr.diversity == 0.5
    assert translated.query.mmr.candidates_limit == 20


def test_single_prefetch_plan_uses_nearest_query() -> None:
    plan = QueryPlan(
        prefetches=(Prefetch(channel=DenseChannel(), limit=80),),
        final_limit=10,
    )
    translated = translate_query_plan(
        plan, query=_make_query(), qdrant_filter=_empty_filter()
    )
    assert translated.prefetch == []
    assert isinstance(translated.query, rest.NearestQuery)
    assert translated.query.nearest == [0.1, 0.2, 0.3]


def test_single_sparse_prefetch_preserves_qdrant_sparse_channel() -> None:
    plan = QueryPlan(
        prefetches=(Prefetch(channel=SparseChannel(vector_field="bm25"), limit=80),),
        final_limit=10,
    )
    translated = translate_query_plan(
        plan, query=_make_query(), qdrant_filter=_empty_filter()
    )

    assert translated.prefetch == []
    assert translated.using == "bm25"
    assert isinstance(translated.query, rest.NearestQuery)
    assert isinstance(translated.query.nearest, rest.SparseVector)


def test_single_prefetch_with_nested_prefetch_is_rejected() -> None:
    plan = QueryPlan(
        prefetches=(
            Prefetch(
                channel=SparseChannel(vector_field="bm25", using_query_vector="bm25"),
                limit=40,
                nested=(Prefetch(channel=DenseChannel(), limit=200),),
            ),
        ),
        final_limit=10,
    )

    with pytest.raises(UnsupportedQueryStage, match="Fuse or MMR rerank"):
        translate_query_plan(
            plan,
            query=_make_query(),
            qdrant_filter=_empty_filter(),
            available_sparse_names={"bm25"},
        )


def test_unavailable_sparse_channel_is_rejected() -> None:
    plan = QueryPlan(
        prefetches=(
            Prefetch(channel=DenseChannel(), limit=80),
            Prefetch(
                channel=SparseChannel(vector_field="bm25", using_query_vector="bm25"),
                limit=80,
            ),
            Prefetch(
                channel=SparseChannel(
                    vector_field="splade", using_query_vector="splade"
                ),
                limit=80,
            ),
        ),
        fuse=PrefetchFusion(),
        final_limit=20,
    )
    with pytest.raises(UnsupportedQueryStage, match="splade"):
        translate_query_plan(
            plan,
            query=_make_query(),
            qdrant_filter=_empty_filter(),
            available_sparse_names={"bm25"},
        )


def test_unavailable_nested_sparse_channel_is_rejected() -> None:
    plan = QueryPlan(
        prefetches=(
            Prefetch(channel=DenseChannel(), limit=80),
            Prefetch(
                channel=SparseChannel(vector_field="bm25", using_query_vector="bm25"),
                limit=80,
                nested=(
                    Prefetch(
                        channel=SparseChannel(
                            vector_field="splade",
                            using_query_vector="splade",
                        ),
                        limit=40,
                    ),
                ),
            ),
        ),
        fuse=PrefetchFusion(),
        final_limit=20,
    )

    with pytest.raises(UnsupportedQueryStage, match="splade"):
        translate_query_plan(
            plan,
            query=_make_query(),
            qdrant_filter=_empty_filter(),
            available_sparse_names={"bm25"},
        )


def test_boost_linear_decay_wraps_base_query_plan_in_formula_query() -> None:
    plan = QueryPlan(
        prefetches=(
            Prefetch(channel=DenseChannel(), limit=80),
            Prefetch(channel=SparseChannel(), limit=80),
        ),
        fuse=PrefetchFusion(kind="dbsf"),
        boost=Boost(
            kind="linear_decay",
            field="freshness",
            params={"scale": 7.0, "midpoint": 0.6},
        ),
        final_limit=20,
    )
    translated = translate_query_plan(
        plan, query=_make_query(), qdrant_filter=_empty_filter()
    )

    assert len(translated.prefetch) == 1
    candidate_stage = translated.prefetch[0]
    assert isinstance(candidate_stage.query, rest.FusionQuery)
    assert candidate_stage.query.fusion == rest.Fusion.DBSF
    nested = cast(list[rest.Prefetch], candidate_stage.prefetch)
    assert [p.using for p in nested] == ["", "bm25"]

    assert isinstance(translated.query, rest.FormulaQuery)
    assert isinstance(translated.query.formula, rest.SumExpression)
    terms = cast(list[object], translated.query.formula.sum)
    assert terms[0] == "$score"
    assert isinstance(terms[1], rest.LinDecayExpression)
    assert terms[1].lin_decay.x == "freshness"
    assert terms[1].lin_decay.scale == 7.0
    assert terms[1].lin_decay.midpoint == 0.6


def test_boost_raw_formula_is_translated_for_qdrant() -> None:
    plan = QueryPlan(
        prefetches=(Prefetch(channel=DenseChannel(), limit=80),),
        boost=Boost(
            kind="raw",
            params={
                "formula": rest.SumExpression(sum=["$score", 1.5]),
                "defaults": {"missing_field": 0.0},
            },
        ),
        final_limit=20,
    )
    translated = translate_query_plan(
        plan, query=_make_query(), qdrant_filter=_empty_filter()
    )

    assert len(translated.prefetch) == 1
    candidate_stage = translated.prefetch[0]
    assert isinstance(candidate_stage.query, rest.NearestQuery)
    assert isinstance(translated.query, rest.FormulaQuery)
    assert isinstance(translated.query.formula, rest.SumExpression)
    assert translated.query.defaults == {"missing_field": 0.0}


def test_boost_rejects_invalid_decay_scale() -> None:
    plan = QueryPlan(
        prefetches=(Prefetch(channel=DenseChannel(), limit=80),),
        boost=Boost(kind="exp_decay", field="freshness", params={"scale": 0.0}),
        final_limit=20,
    )
    with pytest.raises(UnsupportedQueryStage, match="scale"):
        translate_query_plan(plan, query=_make_query(), qdrant_filter=_empty_filter())


def test_nested_prefetch_is_preserved_in_fused_plan() -> None:
    inner = Prefetch(channel=DenseChannel(), limit=200)
    outer = Prefetch(
        channel=SparseChannel(vector_field="bm25", using_query_vector="bm25"),
        limit=20,
        nested=(inner,),
    )
    plan = QueryPlan(
        prefetches=(
            Prefetch(channel=DenseChannel(), limit=80),
            outer,
        ),
        fuse=PrefetchFusion(),
        final_limit=10,
    )
    translated = translate_query_plan(
        plan,
        query=_make_query(),
        qdrant_filter=_empty_filter(),
        available_sparse_names={"bm25"},
    )
    sparse_outer = translated.prefetch[1]
    assert sparse_outer.using == "bm25"
    nested = cast(list[rest.Prefetch], sparse_outer.prefetch)
    assert len(nested) == 1
    assert nested[0].using == ""


def test_adapter_search_consumes_query_plan_when_provided() -> None:
    client = _FakeQdrantClient(existing_names=["docs"], sparse_names=["bm25"])
    plan = QueryPlan(
        prefetches=(
            Prefetch(channel=DenseChannel(), limit=80),
            Prefetch(
                channel=SparseChannel(vector_field="bm25", using_query_vector="bm25"),
                limit=80,
            ),
        ),
        fuse=PrefetchFusion(kind="dbsf"),
        final_limit=20,
    )
    query = _make_query(query_plan=plan)
    asyncio.run(_store_with_fake_client(client).search(query))

    call = client.query_points_calls[0]
    assert call["collection_name"] == "docs"
    assert isinstance(call["query"], rest.FusionQuery)
    assert call["query"].fusion == rest.Fusion.DBSF
    assert [p.using for p in call["prefetch"]] == ["", "bm25"]
    assert call["limit"] == 20


def test_adapter_search_uses_capability_aware_default_query_plan() -> None:
    client = _FakeQdrantClient(existing_names=["docs"], sparse_names=["bm25"])
    query = _make_query(query_plan=None, limit=5)

    asyncio.run(
        _store_with_fake_client(client, sparse_names=frozenset({"bm25"})).search(query)
    )

    call = client.query_points_calls[0]
    assert isinstance(call["query"], rest.FusionQuery)
    assert call["query"].fusion == rest.Fusion.RRF
    assert [p.using for p in call["prefetch"]] == ["", "bm25"]
    assert call["limit"] == 5


def test_direct_qdrant_default_plan_uses_balanced_profile_for_primary_sparse() -> None:
    query = _make_query(query_plan=None, limit=5)

    plan = _default_query_plan_for_available_sparse_channels(
        query=query,
        result_limit=query.limit,
        available_sparse_vector_names={"bm25", "splade"},
    )

    assert plan.search_profile == "balanced"
    assert [
        getattr(prefetch.channel, "vector_field", "")
        for prefetch in plan.prefetches
    ] == ["", "bm25"]


def test_direct_qdrant_default_plan_keeps_custom_sparse_shape_anonymous() -> None:
    query = _make_query(
        query_plan=None,
        limit=5,
        sparse_vector=SparseVector(indices=[], values=[]),
    )

    plan = _default_query_plan_for_available_sparse_channels(
        query=query,
        result_limit=query.limit,
        available_sparse_vector_names={"splade"},
    )

    assert plan.search_profile is None
    assert [
        getattr(prefetch.channel, "vector_field", "")
        for prefetch in plan.prefetches
    ] == ["", "splade"]


def test_adapter_default_query_plan_prefers_primary_sparse_channel() -> None:
    store = _store_with_fake_client(
        _FakeQdrantClient(existing_names=["docs"], sparse_names=["bm25", "splade"]),
        sparse_names=frozenset({"bm25", "splade"}),
    )

    plan = store.default_query_plan(result_limit=5)

    assert plan is not None
    assert [
        getattr(prefetch.channel, "vector_field", "")
        for prefetch in plan.prefetches
    ] == ["", "bm25"]
    assert plan.search_profile == "balanced"


def test_adapter_default_query_plan_uses_available_sparse_channels() -> None:
    store = _store_with_fake_client(
        _FakeQdrantClient(existing_names=["docs"], sparse_names=["splade"]),
        sparse_names=frozenset({"splade"}),
    )

    plan = store.default_query_plan(result_limit=5)

    assert plan is not None
    assert [
        getattr(prefetch.channel, "vector_field", "")
        for prefetch in plan.prefetches
    ] == ["", "splade"]
    assert plan.search_profile is None
