"""Qdrant search execution helpers."""

from __future__ import annotations

from collections.abc import Iterable

from qdrant_client import AsyncQdrantClient

from rag_core.search.planning import (
    DEFAULT_SEARCH_PROFILE,
    QUERY_PLAN_PRESET_DENSE_ONLY,
    default_query_plan,
    query_plan_preset,
    search_profile,
)
from rag_core.search.policy import VectorStorePolicy
from rag_core.search.request_models import SearchQuery
from rag_core.search.query_plan import QueryPlan
from rag_core.search.vector_models import SearchResult

from .qdrant_filters import build_search_filter
from .qdrant_payloads import _point_to_result
from .qdrant_query_plan import translate_query_plan
from .qdrant_shared import _KNOWN_SPARSE_VECTOR_NAMES, _PRIMARY_SPARSE_VECTOR_NAME


async def search_qdrant_points(
    *,
    client: AsyncQdrantClient,
    collection_name: str,
    query: SearchQuery,
    namespace: str,
    policy: VectorStorePolicy,
    available_sparse_vector_names: frozenset[str] | set[str],
) -> list[SearchResult]:
    if query.has_empty_allowlist():
        return []
    qdrant_filter = build_search_filter(
        query=query,
        namespace=namespace,
        policy=policy,
    )
    plan = query.query_plan or _default_query_plan_for_available_sparse_channels(
        query=query,
        result_limit=query.limit,
        available_sparse_vector_names=available_sparse_vector_names,
    )
    if plan is None:
        raise RuntimeError("Qdrant query-plan capabilities did not produce a plan")
    translated = translate_query_plan(
        plan,
        query=query,
        qdrant_filter=qdrant_filter,
        available_sparse_names=available_sparse_vector_names,
    )
    response = await client.query_points(
        collection_name=collection_name,
        prefetch=translated.prefetch or None,
        query=translated.query,
        using=translated.using,
        query_filter=translated.query_filter,
        limit=translated.limit,
        with_payload=True,
    )
    return [_point_to_result(point, policy=policy) for point in response.points]


def _default_query_plan_for_available_sparse_channels(
    *,
    query: SearchQuery,
    result_limit: int,
    available_sparse_vector_names: frozenset[str] | set[str],
) -> QueryPlan:
    query_sparse_names = set(query.all_sparse_vectors())
    return qdrant_default_query_plan_for_sparse_channels(
        result_limit=result_limit,
        sparse_channels=(
            name for name in available_sparse_vector_names if name in query_sparse_names
        ),
    )


def qdrant_default_query_plan_for_sparse_channels(
    *,
    result_limit: int,
    sparse_channels: Iterable[str],
) -> QueryPlan:
    known_sparse_channels = tuple(
        sorted(name for name in sparse_channels if name in _KNOWN_SPARSE_VECTOR_NAMES)
    )
    if _PRIMARY_SPARSE_VECTOR_NAME in known_sparse_channels:
        return search_profile(DEFAULT_SEARCH_PROFILE, limit=result_limit)
    if known_sparse_channels:
        return default_query_plan(
            result_limit=result_limit,
            sparse_channels=known_sparse_channels,
        )
    return query_plan_preset(QUERY_PLAN_PRESET_DENSE_ONLY, limit=result_limit)
