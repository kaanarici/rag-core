"""TurboPuffer search execution helpers."""

from __future__ import annotations

from rag_core.search.policy import VectorStorePolicy
from rag_core.search.query_plan import UnsupportedQueryStage
from rag_core.search.types import SearchQuery, SearchResult, SparseVector

from .memory_query_scoring import reciprocal_rank_fusion
from .turbopuffer_client import TurboPufferNamespace
from .turbopuffer_filters import _search_filter
from .turbopuffer_query_plan import (
    TurboPufferSearchExecution,
    resolve_sparse_query_vector,
    resolve_turbopuffer_search_execution,
)
from .turbopuffer_rows import _required_response_rows, _row_to_result
from .vector_dimensions import validate_query_dense_dimensions

_TURBOPUFFER_SPARSE_FIELD = "sparse_vector"


def _sparse_vector_to_rank_map(sparse_vector: SparseVector) -> dict[str, float]:
    return {
        f"dim{index}": value
        for index, value in zip(
            sparse_vector.indices,
            sparse_vector.values,
            strict=True,
        )
    }


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

    if execution.mode == "dense":
        return await _search_dense(
            namespace_client=namespace_client,
            query=query,
            dense_dimensions=dense_dimensions,
            distance_metric=distance_metric,
            policy=policy,
            filters=filters,
            top_k=execution.final_limit,
        )
    if execution.mode == "sparse_knn":
        return await _search_sparse_knn(
            namespace_client=namespace_client,
            query=query,
            distance_metric=distance_metric,
            policy=policy,
            filters=filters,
            execution=execution,
        )
    return await _search_hybrid_rrf(
        namespace_client=namespace_client,
        query=query,
        dense_dimensions=dense_dimensions,
        distance_metric=distance_metric,
        policy=policy,
        filters=filters,
        execution=execution,
    )


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
        backend="turbopuffer",
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


async def _search_sparse_knn(
    *,
    namespace_client: TurboPufferNamespace,
    query: SearchQuery,
    distance_metric: str,
    policy: VectorStorePolicy,
    filters: object,
    execution: TurboPufferSearchExecution,
) -> list[SearchResult]:
    sparse_vector = resolve_sparse_query_vector(
        query,
        channel_name=execution.sparse_channel or "bm25",
    )
    response = await namespace_client.query(
        rank_by=(
            _TURBOPUFFER_SPARSE_FIELD,
            "SparseKNN",
            _sparse_vector_to_rank_map(sparse_vector),
        ),
        filters=filters,
        top_k=execution.final_limit,
        include_attributes=True,
    )
    return [
        _row_to_result(row, distance_metric=distance_metric, policy=policy)
        for row in _required_response_rows(response, operation="search")
    ]


async def _search_hybrid_rrf(
    *,
    namespace_client: TurboPufferNamespace,
    query: SearchQuery,
    dense_dimensions: int,
    distance_metric: str,
    policy: VectorStorePolicy,
    filters: object,
    execution: TurboPufferSearchExecution,
) -> list[SearchResult]:
    subqueries: list[dict[str, object]] = []
    if execution.dense_limit is not None:
        if not query.dense_vector:
            raise UnsupportedQueryStage(
                "turbopuffer adapter requires a dense vector for hybrid search"
            )
        validate_query_dense_dimensions(
            query.dense_vector,
            dense_dimensions=dense_dimensions,
            backend="turbopuffer",
        )
        subqueries.append(
            {
                "rank_by": ("vector", "ANN", query.dense_vector),
                "filters": filters,
                "top_k": execution.dense_limit,
                "include_attributes": True,
            }
        )

    if execution.sparse_limit is not None:
        lexical_query = (query.lexical_query or "").strip()
        sparse_channel = execution.sparse_channel or "bm25"
        if lexical_query:
            subqueries.append(
                {
                    "rank_by": (policy.text_field, "BM25", lexical_query),
                    "filters": filters,
                    "top_k": execution.sparse_limit,
                    "include_attributes": True,
                }
            )
        else:
            sparse_vector = resolve_sparse_query_vector(
                query,
                channel_name=sparse_channel,
            )
            subqueries.append(
                {
                    "rank_by": (
                        _TURBOPUFFER_SPARSE_FIELD,
                        "SparseKNN",
                        _sparse_vector_to_rank_map(sparse_vector),
                    ),
                    "filters": filters,
                    "top_k": execution.sparse_limit,
                    "include_attributes": True,
                }
            )

    if not subqueries:
        raise UnsupportedQueryStage(
            "turbopuffer hybrid search requires dense and/or sparse prefetches"
        )
    if len(subqueries) == 1:
        response = await namespace_client.query(**subqueries[0])
        rows = _required_response_rows(response, operation="search")
        return [
            _row_to_result(row, distance_metric=distance_metric, policy=policy)
            for row in rows[: execution.final_limit]
        ]

    response = await namespace_client.query(queries=subqueries)
    result_sets = getattr(response, "results", None)
    if result_sets is None and isinstance(response, dict):
        result_sets = response.get("results")
    if not isinstance(result_sets, list):
        raise ValueError("turbopuffer multi-query response missing required results")

    rankings: list[list[str]] = []
    results_by_id: dict[str, SearchResult] = {}
    for result in result_sets:
        rows = _required_response_rows(result, operation="search")
        channel_results = [
            _row_to_result(row, distance_metric=distance_metric, policy=policy)
            for row in rows
        ]
        rankings.append([item.id for item in channel_results])
        for item in channel_results:
            results_by_id[item.id] = item

    fused = reciprocal_rank_fusion(rankings, execution.final_limit)
    return [results_by_id[point_id] for point_id, _ in fused]
