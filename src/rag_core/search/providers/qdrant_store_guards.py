"""Qdrant store request validation helpers."""

from __future__ import annotations

from rag_core.search.planning import validate_query_plan_capabilities
from rag_core.search.query_plan import (
    FUSION_KIND_WEIGHTED_RRF,
    DenseChannel,
    Prefetch,
    QueryPlan,
    UnsupportedQueryStage,
)
from rag_core.search.request_models import DeleteFilter, SearchQuery

from .qdrant_query_plan import validate_qdrant_query_plan_shape
from .qdrant_shared import _DENSE_VECTOR_NAME
from .vector_store_capabilities import (
    QDRANT_VECTOR_STORE_CAPABILITY_SPEC,
    QDRANT_VECTOR_STORE_PROVIDER_SPEC,
)
from .vector_dimensions import validate_query_dense_dimensions


def validate_qdrant_search_request(
    query: SearchQuery,
    *,
    dense_dimensions: int,
) -> str:
    namespace = query.namespace.strip()
    if not namespace:
        raise ValueError("namespace is required for search")
    if _query_requires_dense_vector(query):
        if not query.dense_vector:
            raise ValueError("qdrant dense query vector is required for dense query plans")
        validate_query_dense_dimensions(
            query.dense_vector,
            dense_dimensions=dense_dimensions,
            provider_name=QDRANT_VECTOR_STORE_PROVIDER_SPEC.name,
        )
    if query.query_plan is not None:
        validate_query_plan_capabilities(
            query.query_plan,
            capabilities=QDRANT_VECTOR_STORE_CAPABILITY_SPEC.query_plan,
            provider_name=QDRANT_VECTOR_STORE_PROVIDER_SPEC.name,
        )
        validate_qdrant_query_plan_preflight(query.query_plan)
    return namespace


def validate_qdrant_delete_filter(filter_values: DeleteFilter) -> str:
    namespace = (filter_values.namespace or "").strip()
    if not namespace:
        raise ValueError("namespace is required for delete")
    return namespace


def validate_qdrant_query_plan_preflight(plan: QueryPlan) -> None:
    validate_qdrant_query_plan_shape(plan)
    for prefetch in plan.prefetches:
        _validate_qdrant_prefetch_channels(prefetch)
    prefetch_count = len(plan.prefetches)
    if plan.fuse is not None and prefetch_count < 2:
        raise UnsupportedQueryStage("PrefetchFusion requires at least two prefetches")
    if plan.fuse is not None and plan.fuse.kind == FUSION_KIND_WEIGHTED_RRF:
        if len(plan.fuse.weights) != prefetch_count:
            raise UnsupportedQueryStage(
                "PrefetchFusion(weighted_rrf) requires one weight per prefetch "
                f"(got {len(plan.fuse.weights)} weights for {prefetch_count} prefetches)"
            )


def _validate_qdrant_prefetch_channels(prefetch: Prefetch) -> None:
    if isinstance(prefetch.channel, DenseChannel) and prefetch.channel.vector_field not in {
        "",
        _DENSE_VECTOR_NAME,
    }:
        raise UnsupportedQueryStage(
            "Qdrant adapter supports only the primary dense vector channel"
        )
    for nested in prefetch.nested:
        _validate_qdrant_prefetch_channels(nested)


def _query_requires_dense_vector(query: SearchQuery) -> bool:
    plan = query.query_plan
    if plan is None:
        return True
    return _plan_uses_dense(plan)


def _plan_uses_dense(plan: QueryPlan) -> bool:
    if plan.rerank is not None:
        return True
    return any(_prefetch_uses_dense(prefetch) for prefetch in plan.prefetches)


def _prefetch_uses_dense(prefetch: Prefetch) -> bool:
    if isinstance(prefetch.channel, DenseChannel):
        return True
    return any(_prefetch_uses_dense(nested) for nested in prefetch.nested)
