"""TurboPuffer search execution helpers."""

from __future__ import annotations

from rag_core.search.policy import VectorStorePolicy
from rag_core.search.query_plan import UnsupportedQueryStage
from rag_core.search.types import SearchQuery, SearchResult

from .turbopuffer_client import TurboPufferNamespace
from .turbopuffer_filters import _search_filter
from .turbopuffer_query_plan import _supported_query_plan_limit
from .turbopuffer_rows import _required_response_rows, _row_to_result
from .vector_dimensions import validate_query_dense_dimensions


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
    if not query.dense_vector:
        raise UnsupportedQueryStage(
            "turbopuffer adapter requires a dense vector for base search"
        )
    validate_query_dense_dimensions(
        query.dense_vector,
        dense_dimensions=dense_dimensions,
        backend="turbopuffer",
    )

    top_k = _supported_query_plan_limit(query.query_plan, fallback=query.limit)
    response = await namespace_client.query(
        rank_by=("vector", "ANN", query.dense_vector),
        filters=_search_filter(query=query, namespace=namespace, policy=policy),
        top_k=top_k,
        include_attributes=True,
    )
    return [
        _row_to_result(row, distance_metric=distance_metric, policy=policy)
        for row in _required_response_rows(response, operation="search")
    ]
