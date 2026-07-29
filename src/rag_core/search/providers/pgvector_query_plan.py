from __future__ import annotations

from rag_core.search.query_plan import (
    PRIMARY_DENSE_QUERY_VECTOR,
    DenseChannel,
    QueryPlan,
    UnsupportedQueryStage,
)
from rag_core.search.request_models import SearchQuery


def resolve_pgvector_search_limit(query: SearchQuery) -> int:
    plan = query.query_plan
    if plan is None:
        return query.limit
    return _limit_from_dense_plan(plan)


def validate_pgvector_query_plan(plan: QueryPlan) -> None:
    _limit_from_dense_plan(plan)


def _limit_from_dense_plan(plan: QueryPlan) -> int:
    if plan.rerank is not None:
        raise UnsupportedQueryStage("pgvector adapter cannot honor MMR query plans")
    if plan.boost is not None:
        raise UnsupportedQueryStage("pgvector adapter cannot honor boost query plans")
    if plan.fuse is not None:
        raise UnsupportedQueryStage("pgvector adapter supports dense query plans only")
    if len(plan.prefetches) != 1:
        raise UnsupportedQueryStage("pgvector adapter supports one dense prefetch")
    prefetch = plan.prefetches[0]
    if prefetch.nested:
        raise UnsupportedQueryStage(
            "pgvector adapter cannot honor nested query-plan prefetches"
        )
    if not isinstance(prefetch.channel, DenseChannel):
        raise UnsupportedQueryStage("pgvector adapter supports dense query plans only")
    if prefetch.channel.vector_field or (
        prefetch.channel.using_query_vector != PRIMARY_DENSE_QUERY_VECTOR
    ):
        raise UnsupportedQueryStage(
            "pgvector adapter supports only the primary dense query vector"
        )
    return plan.final_limit
