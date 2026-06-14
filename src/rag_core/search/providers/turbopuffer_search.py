"""TurboPuffer search execution helpers."""

from __future__ import annotations

import asyncio
from dataclasses import replace

from rag_core.search.policy import VectorStorePolicy
from rag_core.search.query_plan import UnsupportedQueryStage
from rag_core.search.request_models import SearchQuery
from rag_core.search.vector_models import SearchResult

from .turbopuffer_client import TurboPufferNamespace
from .turbopuffer_filters import _search_filter
from .turbopuffer_payloads import TURBOPUFFER_BM25_TEXT_FIELD
from .turbopuffer_query_plan import (
    TurboPufferBm25Execution,
    TurboPufferDenseExecution,
    TurboPufferHybridRrfExecution,
    resolve_turbopuffer_search_execution,
)
from .turbopuffer_rows import _required_response_rows, _row_to_result
from .vector_store_capabilities import TURBOPUFFER_VECTOR_STORE_PROVIDER_SPEC
from .vector_dimensions import validate_query_dense_dimensions

_TURBOPUFFER_BM25_DISTANCE_METRIC = "bm25"


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
