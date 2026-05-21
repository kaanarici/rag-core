from __future__ import annotations

from typing import Sequence

from rag_core.search.planning import resolve_prefetch_limit
from rag_core.search.query_plan import (
    DenseChannel,
    QueryPlan,
    SparseChannel,
    UnsupportedQueryStage,
)
from rag_core.search.sparse_channels import (
    KNOWN_SPARSE_CHANNELS,
)
from rag_core.search.types import SearchQuery, SparseVector

from .memory_query_scoring import MemoryPoint
from .memory_query_scoring import rank_dense_points
from .memory_query_scoring import rank_sparse_points
from .memory_query_scoring import reciprocal_rank_fusion
from .memory_query_plan_validation import (
    validate_memory_query_plan,
    validate_memory_sparse_channel,
)


def rank_memory_points(
    query: SearchQuery,
    candidates: Sequence[MemoryPoint],
) -> list[tuple[str, float]]:
    rankings, result_limit = _rankings_for_query(query, candidates)
    return reciprocal_rank_fusion(rankings, result_limit)


def _rankings_for_query(
    query: SearchQuery,
    candidates: Sequence[MemoryPoint],
) -> tuple[list[list[str]], int]:
    plan = query.query_plan
    if plan is None:
        return _default_rankings(query, candidates), query.limit
    return _query_plan_rankings(
        plan, query=query, candidates=candidates
    ), plan.final_limit


def _default_rankings(
    query: SearchQuery,
    candidates: Sequence[MemoryPoint],
) -> list[list[str]]:
    prefetch_limit = resolve_prefetch_limit(result_limit=query.limit)
    rankings: list[list[str]] = []

    dense_ranking = rank_dense_points(query.dense_vector, candidates, prefetch_limit)
    if dense_ranking:
        rankings.append(dense_ranking)

    for name, sparse_vector in query.all_sparse_vectors().items():
        if name not in KNOWN_SPARSE_CHANNELS:
            continue
        sparse_ranking = rank_sparse_points(
            name,
            sparse_vector,
            candidates,
            prefetch_limit,
            fallback_to_primary=False,
        )
        if sparse_ranking:
            rankings.append(sparse_ranking)
    return rankings


def _query_plan_rankings(
    plan: QueryPlan,
    *,
    query: SearchQuery,
    candidates: Sequence[MemoryPoint],
) -> list[list[str]]:
    validate_memory_query_plan(plan)
    sparse_vectors = query.all_sparse_vectors()
    rankings: list[list[str]] = []
    for prefetch in plan.prefetches:
        channel = prefetch.channel
        if isinstance(channel, DenseChannel):
            ranking = rank_dense_points(query.dense_vector, candidates, prefetch.limit)
        elif isinstance(channel, SparseChannel):
            ranking = _rank_sparse_channel(
                channel=channel,
                query_sparse_vectors=sparse_vectors,
                candidates=candidates,
                limit=prefetch.limit,
            )
        else:
            raise UnsupportedQueryStage(
                f"Unknown channel type: {type(channel).__name__}"
            )
        if ranking:
            rankings.append(ranking)
    return rankings


def _rank_sparse_channel(
    *,
    channel: SparseChannel,
    query_sparse_vectors: dict[str, SparseVector],
    candidates: Sequence[MemoryPoint],
    limit: int,
) -> list[str]:
    validate_memory_sparse_channel(channel)
    sparse_vector = query_sparse_vectors.get(channel.using_query_vector)
    if sparse_vector is None:
        raise UnsupportedQueryStage(
            f"SparseChannel({channel.using_query_vector!r}) has no matching sparse query vector"
        )
    return rank_sparse_points(
        channel.vector_field,
        sparse_vector,
        candidates,
        limit,
        fallback_to_primary=False,
    )
