"""TurboPuffer query-plan support checks."""

from __future__ import annotations

from dataclasses import dataclass

from rag_core.search.query_plan import (
    DEFAULT_RRF_K,
    FUSION_KIND_RRF,
    PRIMARY_DENSE_QUERY_VECTOR,
    DenseChannel,
    Prefetch,
    QueryPlan,
    SparseChannel,
    UnsupportedQueryStage,
)
from rag_core.search.request_models import SearchQuery
from rag_core.search.sparse_channels import PRIMARY_SPARSE_CHANNEL


@dataclass(frozen=True)
class TurboPufferDenseExecution:
    final_limit: int
    dense_limit: int


@dataclass(frozen=True)
class TurboPufferBm25Execution:
    final_limit: int
    bm25_limit: int


@dataclass(frozen=True)
class TurboPufferHybridRrfExecution:
    final_limit: int
    dense_limit: int
    bm25_limit: int
    rrf_k: int = DEFAULT_RRF_K


TurboPufferSearchExecution = (
    TurboPufferDenseExecution | TurboPufferBm25Execution | TurboPufferHybridRrfExecution
)


def resolve_turbopuffer_search_execution(
    query: SearchQuery,
) -> TurboPufferSearchExecution:
    plan = query.query_plan
    if plan is None:
        return TurboPufferDenseExecution(
            final_limit=query.limit,
            dense_limit=query.limit,
        )
    return _execution_from_plan(plan)


def _supported_query_plan_limit(plan: QueryPlan | None, *, fallback: int) -> int:
    if plan is None:
        return fallback
    return _execution_from_plan(plan).final_limit


def _execution_from_plan(plan: QueryPlan) -> TurboPufferSearchExecution:
    if plan.rerank is not None:
        raise UnsupportedQueryStage("turbopuffer adapter cannot honor MMR query plans")
    if plan.boost is not None:
        raise UnsupportedQueryStage(
            "turbopuffer adapter cannot honor boost query plans"
        )
    if any(prefetch.nested for prefetch in plan.prefetches):
        raise UnsupportedQueryStage(
            "turbopuffer adapter cannot honor nested query-plan prefetches"
        )

    dense_prefetches: list[tuple[Prefetch, DenseChannel]] = []
    sparse_prefetches: list[tuple[Prefetch, SparseChannel]] = []
    for prefetch in plan.prefetches:
        channel = prefetch.channel
        if isinstance(channel, DenseChannel):
            dense_prefetches.append((prefetch, channel))
        elif isinstance(channel, SparseChannel):
            sparse_prefetches.append((prefetch, channel))

    if len(dense_prefetches) + len(sparse_prefetches) != len(plan.prefetches):
        raise UnsupportedQueryStage(
            "turbopuffer adapter supports only dense and BM25 query-plan channels"
        )

    if not dense_prefetches and not sparse_prefetches:
        raise UnsupportedQueryStage(
            "turbopuffer adapter requires at least one query-plan prefetch"
        )

    if len(dense_prefetches) > 1:
        raise UnsupportedQueryStage(
            "turbopuffer adapter supports at most one dense prefetch"
        )
    if len(sparse_prefetches) > 1:
        raise UnsupportedQueryStage(
            "turbopuffer adapter supports at most one BM25 prefetch"
        )

    if plan.fuse is not None:
        if plan.fuse.kind != FUSION_KIND_RRF:
            raise UnsupportedQueryStage(
                "turbopuffer adapter supports only hybrid RRF fusion"
            )
        if len(dense_prefetches) != 1 or len(sparse_prefetches) != 1:
            raise UnsupportedQueryStage(
                "turbopuffer hybrid RRF requires one dense and one BM25 prefetch"
            )
        dense_prefetch, dense_channel = dense_prefetches[0]
        bm25_prefetch, bm25_channel = sparse_prefetches[0]
        _validate_dense_channel(dense_channel)
        _validate_bm25_channel(bm25_channel)
        _validate_rrf_k(plan.fuse.rrf_k)
        return TurboPufferHybridRrfExecution(
            final_limit=plan.final_limit,
            dense_limit=dense_prefetch.limit,
            bm25_limit=bm25_prefetch.limit,
            rrf_k=plan.fuse.rrf_k,
        )

    if len(plan.prefetches) != 1:
        raise UnsupportedQueryStage(
            "turbopuffer adapter requires RRF fusion for hybrid query plans"
        )
    prefetch = plan.prefetches[0]
    if isinstance(prefetch.channel, DenseChannel):
        _validate_dense_channel(prefetch.channel)
        return TurboPufferDenseExecution(
            final_limit=plan.final_limit,
            dense_limit=prefetch.limit,
        )
    _validate_bm25_channel(prefetch)
    return TurboPufferBm25Execution(
        final_limit=plan.final_limit,
        bm25_limit=prefetch.limit,
    )


def _validate_dense_channel(channel: DenseChannel) -> None:
    if channel.vector_field or channel.using_query_vector != PRIMARY_DENSE_QUERY_VECTOR:
        raise UnsupportedQueryStage(
            "turbopuffer adapter supports only the primary dense query vector"
        )


def _validate_bm25_channel(channel_or_prefetch: SparseChannel | Prefetch) -> None:
    channel = (
        channel_or_prefetch.channel
        if isinstance(channel_or_prefetch, Prefetch)
        else channel_or_prefetch
    )
    if not isinstance(channel, SparseChannel):
        raise UnsupportedQueryStage(
            "turbopuffer BM25 search requires a sparse query-plan channel"
        )
    if (
        channel.vector_field != PRIMARY_SPARSE_CHANNEL
        or channel.using_query_vector != PRIMARY_SPARSE_CHANNEL
    ):
        raise UnsupportedQueryStage(
            "turbopuffer BM25 search supports only the primary BM25 channel"
        )


def _validate_rrf_k(value: int) -> None:
    if value <= 0:
        raise UnsupportedQueryStage("turbopuffer hybrid RRF requires a positive rrf_k")
