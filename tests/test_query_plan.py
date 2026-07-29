"""Tests for the QueryPlan AST and the default_query_plan builder."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from rag_core.retrieval_defaults import DEFAULT_SEARCH_LIMIT
from rag_core.search import (
    Boost,
    DenseChannel,
    Mmr,
    Prefetch,
    PrefetchFusion,
    QueryPlan,
    SparseChannel,
    SparseVector,
    default_query_plan,
    query_plan_preset,
)
from rag_core.search.pipeline_runner import SearchRequest
from rag_core.search.query_plan import DEFAULT_RRF_K
from rag_core.search.sparse_channels import PRIMARY_SPARSE_CHANNEL
from rag_core.search.request_models import (
    SearchQuery,
    SearchSidecarQuery,
)


# ---------------------------------------------------------------------------
# QueryPlan construction


def test_query_plan_requires_at_least_one_prefetch() -> None:
    with pytest.raises(ValueError, match="at least one Prefetch"):
        QueryPlan(prefetches=())


def test_query_plan_with_multiple_prefetches_requires_fuse() -> None:
    prefetches = (
        Prefetch(channel=DenseChannel(), limit=20),
        Prefetch(channel=SparseChannel(), limit=20),
    )
    with pytest.raises(ValueError, match="PrefetchFusion stage"):
        QueryPlan(prefetches=prefetches)


def test_query_plan_single_prefetch_does_not_require_fuse() -> None:
    plan = QueryPlan(
        prefetches=(Prefetch(channel=DenseChannel(), limit=20),),
        final_limit=10,
    )
    assert plan.fuse is None


def test_query_plan_single_prefetch_rejects_fuse() -> None:
    with pytest.raises(ValueError, match="at least two Prefetches"):
        QueryPlan(
            prefetches=(Prefetch(channel=DenseChannel(), limit=20),),
            fuse=PrefetchFusion(kind="dbsf"),
            final_limit=10,
        )


def test_query_plan_final_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="final_limit"):
        QueryPlan(
            prefetches=(Prefetch(channel=DenseChannel(), limit=20),),
            final_limit=0,
        )


def test_prefetch_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="Prefetch.limit"):
        Prefetch(channel=DenseChannel(), limit=0)


def test_search_request_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="SearchRequest.limit"):
        SearchRequest(query="q", collections=["c"], namespace="n", limit=0)


def test_search_request_query_must_be_non_empty() -> None:
    with pytest.raises(ValueError, match="SearchRequest.query"):
        SearchRequest(query=" ", collections=["c"], namespace="n")


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: SearchRequest(query="q", collections=[" "], namespace="n"),
            "SearchRequest.collections",
        ),
        (
            lambda: SearchRequest(
                query="q",
                collections=["c"],
                namespace="n",
                document_ids=[" "],
            ),
            "SearchRequest.document_ids",
        ),
        (
            lambda: SearchQuery(
                dense_vector=[0.1],
                sparse_vector=SparseVector(indices=[], values=[]),
                namespace="n",
                collections=[" "],
            ),
            "SearchQuery.collections",
        ),
        (
            lambda: SearchSidecarQuery(query="q", namespace="n", collections=[" "]),
            "SearchSidecarQuery.collections",
        ),
        (
            lambda: SearchSidecarQuery(query=" ", namespace="n", collections=["c"]),
            "SearchSidecarQuery.query",
        ),
    ],
)
def test_search_requests_reject_blank_scope_values(
    factory: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_search_query_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="SearchQuery.limit"):
        SearchQuery(
            dense_vector=[0.1],
            sparse_vector=SparseVector(indices=[], values=[]),
            namespace="n",
            collections=["c"],
            limit=0,
        )


def test_search_sidecar_query_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="SearchSidecarQuery.limit"):
        SearchSidecarQuery(query="q", namespace="n", collections=["c"], limit=0)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: SearchRequest(query="q", collections=["c"], namespace=" "),
            "SearchRequest.namespace",
        ),
        (
            lambda: SearchQuery(
                dense_vector=[0.1],
                sparse_vector=SparseVector(indices=[], values=[]),
                namespace=" ",
                collections=["c"],
            ),
            "SearchQuery.namespace",
        ),
        (
            lambda: SearchSidecarQuery(query="q", namespace=" ", collections=["c"]),
            "SearchSidecarQuery.namespace",
        ),
    ],
)
def test_search_requests_reject_blank_namespaces(
    factory: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


@pytest.mark.parametrize("limit", [True, False])
def test_prefetch_limit_rejects_boolean_values(limit: bool) -> None:
    with pytest.raises(ValueError, match="Prefetch.limit"):
        Prefetch(channel=DenseChannel(), limit=limit)


def test_prefetch_supports_nested_prefetch() -> None:
    inner = Prefetch(channel=DenseChannel(), limit=200)
    outer = Prefetch(channel=SparseChannel(), limit=20, nested=(inner,))
    assert outer.nested == (inner,)


def test_query_plan_is_frozen() -> None:
    plan = default_query_plan(result_limit=10)
    with pytest.raises(Exception):
        setattr(plan, "final_limit", 99)


def test_query_plan_can_carry_rerank_and_boost() -> None:
    plan = QueryPlan(
        prefetches=(
            Prefetch(channel=DenseChannel(), limit=80),
            Prefetch(channel=SparseChannel(), limit=80),
        ),
        fuse=PrefetchFusion(),
        rerank=Mmr(diversity=0.5, limit=20),
        boost=Boost(kind="linear_decay", field="ts"),
        final_limit=20,
    )
    assert plan.rerank is not None
    assert plan.boost is not None


# ---------------------------------------------------------------------------
# PrefetchFusion / Mmr / Boost validation


def test_fuse_default_is_rrf_with_k60() -> None:
    fuse = PrefetchFusion()
    assert fuse.kind == "rrf"
    assert fuse.rrf_k == DEFAULT_RRF_K
    assert fuse.weights == ()


def test_fuse_weighted_rrf_requires_weights_and_accepts_them() -> None:
    with pytest.raises(ValueError, match="requires weights"):
        PrefetchFusion(kind="weighted_rrf")
    fuse = PrefetchFusion(kind="weighted_rrf", weights=(2.0, 1.0))
    assert fuse.weights == (2.0, 1.0)


def test_fuse_dbsf_rejects_weights() -> None:
    with pytest.raises(ValueError, match="does not support weights"):
        PrefetchFusion(kind="dbsf", weights=(1.0, 1.0))


def test_fuse_plain_rrf_rejects_weights() -> None:
    with pytest.raises(ValueError, match="does not support weights"):
        PrefetchFusion(kind="rrf", weights=(1.0, 1.0))


@pytest.mark.parametrize("diversity", [0.0, 1.0, -0.1])
def test_mmr_diversity_must_be_in_open_interval(diversity: float) -> None:
    with pytest.raises(ValueError, match="diversity"):
        Mmr(diversity=diversity, limit=10)


def test_mmr_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="limit"):
        Mmr(diversity=0.5, limit=0)


@pytest.mark.parametrize("limit", [True, False])
def test_mmr_limit_rejects_boolean_values(limit: bool) -> None:
    with pytest.raises(ValueError, match="Mmr.limit"):
        Mmr(diversity=0.5, limit=limit)


def test_mmr_constructible_with_valid_params() -> None:
    mmr = Mmr(diversity=0.5, limit=10)
    assert mmr.diversity == 0.5
    assert mmr.limit == 10


def test_boost_decay_requires_field_but_raw_does_not() -> None:
    with pytest.raises(ValueError, match="requires a field"):
        Boost(kind="linear_decay")

    raw = Boost(kind="raw", params={"formula": "score * 2"})
    assert raw.field == ""
    assert raw.params == {"formula": "score * 2"}

    decay = Boost(
        kind="exp_decay",
        field="created_at",
        params={"target": 1700000000.0, "scale": 86400.0},
    )
    assert decay.kind == "exp_decay"
    assert decay.field == "created_at"


# ---------------------------------------------------------------------------
# default_query_plan builder


def test_default_query_plan_is_dense_plus_primary_sparse_with_rrf60() -> None:
    plan = default_query_plan(result_limit=20)
    assert plan.final_limit == 20
    assert len(plan.prefetches) == 2
    assert isinstance(plan.prefetches[0].channel, DenseChannel)
    sparse = plan.prefetches[1].channel
    assert isinstance(sparse, SparseChannel)
    assert sparse.vector_field == PRIMARY_SPARSE_CHANNEL
    assert plan.fuse is not None
    assert plan.fuse.kind == "rrf"
    assert plan.fuse.rrf_k == DEFAULT_RRF_K
    assert plan.prefetches[0].limit == plan.prefetches[1].limit == 80


def test_query_plan_defaults_use_named_search_limit() -> None:
    plan = default_query_plan()
    bare_plan = QueryPlan(prefetches=(Prefetch(channel=DenseChannel(), limit=10),))

    assert plan.final_limit == DEFAULT_SEARCH_LIMIT
    assert bare_plan.final_limit == DEFAULT_SEARCH_LIMIT


def test_default_query_plan_supports_multiple_sparse_channels() -> None:
    plan = default_query_plan(result_limit=10, sparse_channels=("bm25", "splade"))
    assert len(plan.prefetches) == 3
    sparse_fields = [
        p.channel.vector_field
        for p in plan.prefetches
        if isinstance(p.channel, SparseChannel)
    ]
    assert sparse_fields == ["bm25", "splade"]


def test_default_query_plan_supports_dbsf_fusion() -> None:
    plan = default_query_plan(result_limit=20, fusion="dbsf")
    assert plan.fuse is not None
    assert plan.fuse.kind == "dbsf"


def test_default_query_plan_propagates_explicit_prefetch_limit() -> None:
    plan = default_query_plan(result_limit=20, prefetch_limit=100)
    for prefetch in plan.prefetches:
        assert prefetch.limit == 100


def test_mmr_profile_keeps_candidate_limit_wider_than_final_limit() -> None:
    plan = query_plan_preset("hybrid_with_mmr", limit=5)
    assert plan.rerank is not None
    assert plan.final_limit == 5
    assert plan.rerank.limit == 20


def test_query_plan_final_limit_rejects_boolean_values() -> None:
    with pytest.raises(ValueError, match="final_limit"):
        QueryPlan(
            prefetches=(Prefetch(channel=DenseChannel(), limit=20),),
            final_limit=True,
        )


@pytest.mark.parametrize("rrf_k", [True, False])
def test_prefetch_fusion_rrf_k_rejects_boolean_values(rrf_k: bool) -> None:
    with pytest.raises(ValueError, match="PrefetchFusion.rrf_k"):
        PrefetchFusion(rrf_k=rrf_k)


@pytest.mark.parametrize(
    ("final_limit", "requested", "expected"),
    [
        (10, None, 40),  # 4x multiplier
        (3, None, 20),  # floor
        (50, None, 200),  # cap
        # Regression: a large final_limit must never fall below itself. The
        # multiplier pool caps at 200, so the pool must floor at final_limit or
        # a rerank query for >200 results would deliver fewer than requested.
        (250, None, 250),
        (500, None, 500),
        (10, 15, 15),  # explicit requested honored
        (10, 5, 10),  # requested below final_limit floors at final_limit
    ],
)
def test_rerank_candidate_pool_limit_never_below_final_limit(
    final_limit: int, requested: int | None, expected: int
) -> None:
    from rag_core.search.query_plan_presets import rerank_candidate_pool_limit

    assert (
        rerank_candidate_pool_limit(final_limit=final_limit, requested=requested)
        == expected
    )
    assert (
        rerank_candidate_pool_limit(final_limit=final_limit, requested=requested)
        >= final_limit
    )
