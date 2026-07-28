"""Reference contract for the "basic" QueryPlan subset.

Adapters that don't expose a vendor query API (in-memory, pgvector, Pinecone) are
expected to honor a minimal subset of QueryPlan: dense + sparse prefetches plus
a single PrefetchFusion(rrf|dbsf) stage. Everything else must raise
:class:`UnsupportedQueryStage`. The executor below is the canonical reference
this contract exists to document.
"""

from __future__ import annotations

import pytest

from rag_core.search.query_plan import (
    Boost,
    DenseChannel,
    Mmr,
    Prefetch,
    PrefetchFusion,
    QueryPlan,
    SparseChannel,
    UnsupportedQueryStage,
)


def rrf_fuse(rank_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Textbook RRF: score = sum(1 / (k + rank)) over all rank lists."""
    scores: dict[str, float] = {}
    for rank_list in rank_lists:
        for rank, point_id in enumerate(rank_list):
            scores[point_id] = scores.get(point_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def execute_basic_plan(
    plan: QueryPlan,
    *,
    rank_lists_by_channel: dict[str, list[str]],
) -> list[tuple[str, float]]:
    if plan.boost is not None:
        raise UnsupportedQueryStage("Boost is not supported by basic adapters")
    if plan.rerank is not None:
        if isinstance(plan.rerank, Mmr):
            raise UnsupportedQueryStage("MMR rerank requires a vendor query API")
        raise UnsupportedQueryStage(f"Unknown rerank: {type(plan.rerank).__name__}")

    rank_lists: list[list[str]] = []
    for prefetch in plan.prefetches:
        if prefetch.nested:
            raise UnsupportedQueryStage("Nested prefetch requires a vendor query API")
        if isinstance(prefetch.channel, DenseChannel):
            channel_key = "dense"
        else:
            assert isinstance(prefetch.channel, SparseChannel)
            channel_key = prefetch.channel.vector_field
        rank_lists.append(rank_lists_by_channel.get(channel_key, []))

    if len(rank_lists) == 1:
        return [(pid, 1.0 / (60 + i + 1)) for i, pid in enumerate(rank_lists[0])][
            : plan.final_limit
        ]

    if plan.fuse is None:
        raise ValueError("Multiple prefetches require a Fuse stage")
    if plan.fuse.kind == "weighted_rrf":
        raise UnsupportedQueryStage("weighted_rrf not implemented in basic adapter")
    if plan.fuse.kind == "dbsf":
        raise UnsupportedQueryStage("dbsf not implemented in basic adapter")
    return rrf_fuse(rank_lists, k=plan.fuse.rrf_k)[: plan.final_limit]


# ---------------------------------------------------------------------------
# Supported plans execute


def test_basic_adapter_executes_default_two_channel_plan() -> None:
    plan = QueryPlan(
        prefetches=(
            Prefetch(channel=DenseChannel(), limit=80),
            Prefetch(channel=SparseChannel(), limit=80),
        ),
        fuse=PrefetchFusion(),
        final_limit=20,
    )
    results = execute_basic_plan(
        plan,
        rank_lists_by_channel={
            "dense": ["a", "b", "c"],
            "bm25": ["b", "a", "d"],
        },
    )
    point_ids = [r[0] for r in results]
    assert point_ids[0] in ("a", "b")
    assert set(point_ids) == {"a", "b", "c", "d"}


def test_basic_adapter_single_channel_plan_uses_inverse_rank() -> None:
    plan = QueryPlan(
        prefetches=(Prefetch(channel=DenseChannel(), limit=20),),
        final_limit=10,
    )
    results = execute_basic_plan(plan, rank_lists_by_channel={"dense": ["a", "b", "c"]})
    assert [r[0] for r in results] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Unsupported stages raise


@pytest.mark.parametrize(
    "plan_kwargs, match",
    [
        (
            {
                "rerank": Mmr(diversity=0.5, limit=10),
                "fuse": PrefetchFusion(),
            },
            "MMR",
        ),
        (
            {"fuse": PrefetchFusion(kind="weighted_rrf", weights=(2.0, 1.0))},
            "weighted_rrf",
        ),
        ({"fuse": PrefetchFusion(kind="dbsf")}, "dbsf"),
    ],
)
def test_basic_adapter_refuses_unsupported_stages(
    plan_kwargs: dict[str, object], match: str
) -> None:
    plan = QueryPlan(
        prefetches=(
            Prefetch(channel=DenseChannel(), limit=80),
            Prefetch(channel=SparseChannel(), limit=80),
        ),
        final_limit=20,
        **plan_kwargs,  # type: ignore[arg-type]
    )
    with pytest.raises(UnsupportedQueryStage, match=match):
        execute_basic_plan(
            plan, rank_lists_by_channel={"dense": [], "bm25": []}
        )


def test_basic_adapter_refuses_nested_prefetch() -> None:
    inner = Prefetch(channel=DenseChannel(), limit=200)
    outer = Prefetch(channel=SparseChannel(), limit=20, nested=(inner,))
    plan = QueryPlan(prefetches=(outer,), final_limit=10)
    with pytest.raises(UnsupportedQueryStage, match="Nested prefetch"):
        execute_basic_plan(plan, rank_lists_by_channel={})


def test_basic_adapter_refuses_boost() -> None:
    plan = QueryPlan(
        prefetches=(Prefetch(channel=DenseChannel(), limit=20),),
        boost=Boost(kind="linear_decay", field="ts"),
        final_limit=10,
    )
    with pytest.raises(UnsupportedQueryStage, match="Boost"):
        execute_basic_plan(plan, rank_lists_by_channel={})


# ---------------------------------------------------------------------------
# RRF fusion math


def test_rrf_fuse_textbook_formula_and_configurable_k() -> None:
    rank_lists = [["a", "b", "c"], ["b", "a", "c"]]
    results = rrf_fuse(rank_lists, k=60)
    by_id = dict(results)
    assert by_id["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert by_id["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert by_id["c"] == pytest.approx(1 / 63 + 1 / 63)

    # Smaller k boosts top-ranked items more aggressively.
    results_60 = rrf_fuse([["a"], ["a"]], k=60)
    results_2 = rrf_fuse([["a"], ["a"]], k=2)
    assert results_2[0][1] > results_60[0][1]


def test_rrf_fuse_handles_empty_and_disjoint_lists() -> None:
    assert rrf_fuse([], k=60) == []
    assert rrf_fuse([[]], k=60) == []
    disjoint = rrf_fuse([["a", "b"], ["c", "d"]], k=60)
    assert {p for p, _ in disjoint} == {"a", "b", "c", "d"}
    assert disjoint[0][0] in ("a", "c")
