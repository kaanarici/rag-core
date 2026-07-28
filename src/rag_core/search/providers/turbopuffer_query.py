"""TurboPuffer query planning, search, and document-lookup helpers."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace

from rag_core.search.document_records import (
    resolve_document_id_from_payload,
    stored_document_record_from_payload,
    validate_document_lookup_inputs,
)
from rag_core.search.policy import VectorStorePolicy
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
from rag_core.search.request_models import SearchQuery, StoredDocumentRecord
from rag_core.search.sparse_channels import PRIMARY_SPARSE_CHANNEL
from rag_core.search.vector_models import SearchResult

from .turbopuffer_client import TurboPufferNamespace
from .turbopuffer_payloads import (
    TURBOPUFFER_BM25_TEXT_FIELD,
    _document_lookup_filter,
    _required_response_rows,
    _row_payload,
    _row_to_result,
    _search_filter,
)
from .vector_dimensions import validate_query_dense_dimensions
from .vector_store_capabilities import TURBOPUFFER_VECTOR_STORE_PROVIDER_SPEC

_TURBOPUFFER_BM25_DISTANCE_METRIC = "bm25"


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


async def search_turbopuffer_points(
    *,
    namespace_client: TurboPufferNamespace,
    query: SearchQuery,
    dense_dimensions: int,
    distance_metric: str,
    policy: VectorStorePolicy,
) -> list[SearchResult]:
    namespace = query.namespace.strip()
    if not namespace:
        raise ValueError("namespace is required for search")
    if query.has_empty_allowlist():
        return []

    execution = resolve_turbopuffer_search_execution(query)
    filters = _search_filter(query=query, namespace=namespace, policy=policy)

    if isinstance(execution, TurboPufferDenseExecution):
        results = await _search_dense(
            namespace_client=namespace_client,
            query=query,
            dense_dimensions=dense_dimensions,
            distance_metric=distance_metric,
            policy=policy,
            filters=filters,
            top_k=execution.dense_limit,
        )
        return results[: execution.final_limit]
    if isinstance(execution, TurboPufferBm25Execution):
        results = await _search_bm25(
            namespace_client=namespace_client,
            query=query,
            policy=policy,
            filters=filters,
            top_k=execution.bm25_limit,
        )
        return results[: execution.final_limit]
    if isinstance(execution, TurboPufferHybridRrfExecution):
        dense_results, bm25_results = await asyncio.gather(
            _search_dense(
                namespace_client=namespace_client,
                query=query,
                dense_dimensions=dense_dimensions,
                distance_metric=distance_metric,
                policy=policy,
                filters=filters,
                top_k=execution.dense_limit,
            ),
            _search_bm25(
                namespace_client=namespace_client,
                query=query,
                policy=policy,
                filters=filters,
                top_k=execution.bm25_limit,
            ),
        )
        return _fuse_rrf(
            [dense_results, bm25_results],
            final_limit=execution.final_limit,
            rrf_k=execution.rrf_k,
        )
    raise UnsupportedQueryStage("unsupported turbopuffer search execution")


async def _search_dense(
    *,
    namespace_client: TurboPufferNamespace,
    query: SearchQuery,
    dense_dimensions: int,
    distance_metric: str,
    policy: VectorStorePolicy,
    filters: object,
    top_k: int,
) -> list[SearchResult]:
    if not query.dense_vector:
        raise UnsupportedQueryStage(
            "turbopuffer adapter requires a dense vector for dense search"
        )
    validate_query_dense_dimensions(
        query.dense_vector,
        dense_dimensions=dense_dimensions,
        provider_name=TURBOPUFFER_VECTOR_STORE_PROVIDER_SPEC.name,
    )
    response = await namespace_client.query(
        rank_by=("vector", "ANN", query.dense_vector),
        filters=filters,
        top_k=top_k,
        include_attributes=True,
    )
    return [
        _row_to_result(row, distance_metric=distance_metric, policy=policy)
        for row in _required_response_rows(response, operation="search")
    ]


async def _search_bm25(
    *,
    namespace_client: TurboPufferNamespace,
    query: SearchQuery,
    policy: VectorStorePolicy,
    filters: object,
    top_k: int,
) -> list[SearchResult]:
    lexical_query = (query.lexical_query or "").strip()
    if not lexical_query:
        raise UnsupportedQueryStage(
            "turbopuffer BM25 search requires SearchQuery.lexical_query"
        )
    response = await namespace_client.query(
        rank_by=(TURBOPUFFER_BM25_TEXT_FIELD, "BM25", lexical_query),
        filters=filters,
        top_k=top_k,
        include_attributes=True,
    )
    return [
        _row_to_result(
            row,
            distance_metric=_TURBOPUFFER_BM25_DISTANCE_METRIC,
            policy=policy,
        )
        for row in _required_response_rows(response, operation="search")
    ]


def _fuse_rrf(
    result_lists: list[list[SearchResult]],
    *,
    final_limit: int,
    rrf_k: int,
) -> list[SearchResult]:
    scores: dict[str, float] = {}
    first_seen: dict[str, tuple[int, int, str]] = {}
    kept: dict[str, SearchResult] = {}
    for list_index, results in enumerate(result_lists):
        seen_in_list: set[str] = set()
        for rank, result in enumerate(results, start=1):
            if result.id in seen_in_list:
                continue
            seen_in_list.add(result.id)
            scores[result.id] = scores.get(result.id, 0.0) + 1.0 / (rrf_k + rank)
            if result.id not in kept:
                kept[result.id] = result
                first_seen[result.id] = (list_index, rank, result.id)
    ordered = sorted(
        kept,
        key=lambda result_id: (-scores[result_id], first_seen[result_id]),
    )
    return [
        replace(kept[result_id], score=scores[result_id])
        for result_id in ordered[:final_limit]
    ]


async def get_turbopuffer_document_record(
    *,
    namespace_client: TurboPufferNamespace,
    namespace: str,
    collection: str,
    document_id: str | None,
    document_key: str | None,
    policy: VectorStorePolicy,
) -> StoredDocumentRecord | None:
    namespace_scoped, collection_scoped = validate_document_lookup_inputs(
        namespace=namespace,
        collection=collection,
        document_id=document_id,
        document_key=document_key,
    )
    filters = _document_lookup_filter(
        namespace=namespace_scoped,
        collection=collection_scoped,
        document_id=document_id,
        document_key=document_key,
        policy=policy,
    )
    lookup_response = await namespace_client.query(
        rank_by=("id", "asc"),
        filters=filters,
        limit=1,
        include_attributes=True,
    )
    rows = _required_response_rows(lookup_response, operation="document lookup")
    if not rows:
        return None
    sample = _row_payload(rows[0])
    resolved_document_id = resolve_document_id_from_payload(
        payload=sample,
        document_id_field=policy.document_id_field,
        fallback_document_id=document_id,
        invalid_message=(
            "turbopuffer document lookup returned invalid string field: "
            f"{policy.document_id_field}"
        ),
        reject_blank=True,
    )
    if not resolved_document_id:
        return None
    count_response = await namespace_client.query(
        filters=_document_lookup_filter(
            namespace=namespace_scoped,
            collection=collection_scoped,
            document_id=resolved_document_id,
            document_key=None,
            policy=policy,
        ),
        limit=1,
        aggregate_by={"chunk_count": ("Count",)},
    )

    return stored_document_record_from_payload(
        payload=sample,
        namespace=namespace_scoped,
        collection=collection_scoped,
        document_id=resolved_document_id,
        chunk_count=_response_aggregation_int(count_response, "chunk_count"),
        policy=policy,
        invalid_field_message=(
            "turbopuffer document lookup returned invalid string field: {field}"
        ),
    )


def _response_aggregation_int(response: object, key: str) -> int:
    aggregations = getattr(response, "aggregations", None)
    if not isinstance(aggregations, Mapping):
        raise ValueError(
            "turbopuffer document lookup missing required aggregation: %s" % key
        )
    if key not in aggregations:
        raise ValueError(
            "turbopuffer document lookup missing required aggregation: %s" % key
        )
    value = aggregations.get(key)
    parsed = _parse_non_negative_int(value)
    if parsed is None:
        raise ValueError(
            "turbopuffer document lookup returned invalid aggregation: %s" % key
        )
    return parsed


def _parse_non_negative_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None
        parsed = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped.isdecimal():
            return None
        parsed = int(stripped)
    else:
        return None
    if parsed < 0:
        return None
    return parsed
