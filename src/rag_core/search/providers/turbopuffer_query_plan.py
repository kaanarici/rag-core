"""TurboPuffer query-plan support checks."""

from __future__ import annotations

from rag_core.search.query_plan import DenseChannel, QueryPlan, UnsupportedQueryStage


def _supported_query_plan_limit(plan: QueryPlan | None, *, fallback: int) -> int:
    if plan is None:
        return fallback
    if plan.fuse is not None:
        raise UnsupportedQueryStage(
            "turbopuffer adapter supports only single dense-channel query plans"
        )
    if plan.rerank is not None:
        raise UnsupportedQueryStage("turbopuffer adapter cannot honor MMR query plans")
    if plan.boost is not None:
        raise UnsupportedQueryStage(
            "turbopuffer adapter cannot honor boost query plans"
        )
    if len(plan.prefetches) != 1:
        raise UnsupportedQueryStage(
            "turbopuffer adapter supports only single dense-channel query plans"
        )
    prefetch = plan.prefetches[0]
    if prefetch.nested:
        raise UnsupportedQueryStage(
            "turbopuffer adapter cannot honor nested query-plan prefetches"
        )
    channel = prefetch.channel
    if not isinstance(channel, DenseChannel):
        raise UnsupportedQueryStage(
            "turbopuffer adapter supports only dense-channel query plans"
        )
    if channel.vector_field or channel.using_query_vector != "primary":
        raise UnsupportedQueryStage(
            "turbopuffer adapter supports only the primary dense query vector"
        )
    return plan.final_limit
