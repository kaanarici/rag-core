"""TurboPuffer query-plan support checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rag_core.search.query_plan import (
    DenseChannel,
    QueryPlan,
    SparseChannel,
    UnsupportedQueryStage,
)
from rag_core.search.sparse_channels import KNOWN_SPARSE_CHANNELS, PRIMARY_SPARSE_CHANNEL
from rag_core.search.types import SearchQuery, SparseVector


@dataclass(frozen=True)
class TurboPufferSearchExecution:
    final_limit: int
    mode: Literal["dense", "hybrid_rrf", "sparse_knn"]
    dense_limit: int | None = None
    sparse_limit: int | None = None
    sparse_channel: str | None = None


def resolve_turbopuffer_search_execution(
    query: SearchQuery,
) -> TurboPufferSearchExecution:
    plan = query.query_plan
    if plan is None:
        return TurboPufferSearchExecution(final_limit=query.limit, mode="dense")
    return _execution_from_plan(plan, query=query)


def _supported_query_plan_limit(plan: QueryPlan | None, *, fallback: int) -> int:
    if plan is None:
        return fallback
    return _execution_from_plan(plan).final_limit


def _execution_from_plan(plan: QueryPlan, *, query: SearchQuery) -> TurboPufferSearchExecution:
    if plan.rerank is not None:
        raise UnsupportedQueryStage("turbopuffer adapter cannot honor MMR query plans")
    if plan.boost is not None:
        raise UnsupportedQueryStage(
            "turbopuffer adapter cannot honor boost query plans"
        )
    if plan.fuse is not None and plan.fuse.kind != "rrf":
        raise UnsupportedQueryStage(
            "turbopuffer adapter supports only reciprocal-rank fusion"
        )
    if any(prefetch.nested for prefetch in plan.prefetches):
        raise UnsupportedQueryStage(
            "turbopuffer adapter cannot honor nested query-plan prefetches"
        )

    dense_prefetches = [
        prefetch
        for prefetch in plan.prefetches
        if isinstance(prefetch.channel, DenseChannel)
    ]
    sparse_prefetches = [
        prefetch
        for prefetch in plan.prefetches
        if isinstance(prefetch.channel, SparseChannel)
    ]
    if len(dense_prefetches) + len(sparse_prefetches) != len(plan.prefetches):
        raise UnsupportedQueryStage(
            "turbopuffer adapter supports only dense and sparse query-plan channels"
        )

    if not dense_prefetches and not sparse_prefetches:
        raise UnsupportedQueryStage(
            "turbopuffer adapter requires at least one query-plan prefetch"
        )

    if len(dense_prefetches) > 1 or len(sparse_prefetches) > 1:
        raise UnsupportedQueryStage(
            "turbopuffer adapter supports at most one dense and one sparse prefetch"
        )

    dense_limit = dense_prefetches[0].limit if dense_prefetches else None
    sparse_limit = sparse_prefetches[0].limit if sparse_prefetches else None
    sparse_channel = (
        sparse_prefetches[0].channel.using_query_vector if sparse_prefetches else None
    )

    if dense_prefetches and sparse_prefetches:
        if plan.fuse is None:
            raise UnsupportedQueryStage(
                "turbopuffer adapter requires fusion for hybrid query plans"
            )
        _validate_dense_channel(dense_prefetches[0].channel)
        _validate_sparse_channel(sparse_prefetches[0].channel)
        return TurboPufferSearchExecution(
            final_limit=plan.final_limit,
            mode="hybrid_rrf",
            dense_limit=dense_limit,
            sparse_limit=sparse_limit,
            sparse_channel=sparse_channel,
        )

    if dense_prefetches:
        if plan.fuse is not None:
            raise UnsupportedQueryStage(
                "turbopuffer adapter supports only single dense-channel query plans"
            )
        _validate_dense_channel(dense_prefetches[0].channel)
        return TurboPufferSearchExecution(
            final_limit=plan.final_limit,
            mode="dense",
            dense_limit=dense_limit,
        )

    if plan.fuse is not None:
        raise UnsupportedQueryStage(
            "turbopuffer adapter supports only single sparse-channel query plans"
        )
    _validate_sparse_channel(sparse_prefetches[0].channel)
    return TurboPufferSearchExecution(
        final_limit=plan.final_limit,
        mode="sparse_knn",
        sparse_limit=sparse_limit,
        sparse_channel=sparse_channel or PRIMARY_SPARSE_CHANNEL,
    )


def _validate_dense_channel(channel: DenseChannel) -> None:
    if channel.vector_field or channel.using_query_vector != "primary":
        raise UnsupportedQueryStage(
            "turbopuffer adapter supports only the primary dense query vector"
        )


def _validate_sparse_channel(channel: SparseChannel) -> None:
    if channel.vector_field and channel.vector_field not in KNOWN_SPARSE_CHANNELS:
        raise UnsupportedQueryStage(
            "turbopuffer adapter does not support sparse vector field "
            f"{channel.vector_field!r}"
        )


def resolve_sparse_query_vector(
    query: SearchQuery,
    *,
    channel_name: str,
) -> SparseVector:
    sparse_vectors = query.all_sparse_vectors()
    sparse_vector = sparse_vectors.get(channel_name)
    if sparse_vector is None:
        raise UnsupportedQueryStage(
            f"SparseChannel({channel_name!r}) has no matching sparse query vector"
        )
    if not sparse_vector.indices:
        raise UnsupportedQueryStage(
            "turbopuffer sparse retrieval requires a non-empty sparse query vector"
        )
    return sparse_vector
