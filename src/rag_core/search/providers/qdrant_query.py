"""Translate the three supported retrieval modes to Qdrant queries."""

from __future__ import annotations
from qdrant_client import AsyncQdrantClient, models as rest
from rag_core.search.policy import VectorStorePolicy
from rag_core.search.request_models import DeleteFilter, SearchQuery
from rag_core.search.document_versions import Candidate, candidate
from .qdrant_filters import build_search_filter
from .qdrant_payloads import (
    _DENSE_VECTOR_NAME,
    _PRIMARY_SPARSE_VECTOR_NAME,
    _point_to_result,
)
from .vector_dimensions import validate_query_dense_dimensions


def validate_qdrant_search_request(query: SearchQuery, *, dense_dimensions: int) -> str:
    if query.mode != "lexical":
        validate_query_dense_dimensions(
            query.dense_vector,
            dense_dimensions=dense_dimensions,
            provider_name="qdrant",
        )
    return query.tenant_id.strip()


def validate_qdrant_delete_filter(filter_values: DeleteFilter) -> str:
    tenant = (filter_values.tenant_id or "").strip()
    if not tenant:
        raise ValueError("tenant_id is required for delete")
    return tenant


async def search_qdrant_points(
    *,
    client: AsyncQdrantClient,
    collection_name: str,
    query: SearchQuery,
    tenant_id: str,
    policy: VectorStorePolicy,
    available_sparse_vector_names: frozenset[str] | set[str],
    consistency: rest.ReadConsistency | None = None,
) -> list[Candidate]:
    if query.has_empty_allowlist():
        return []
    scope = build_search_filter(query=query, tenant_id=tenant_id, policy=policy)
    using: str | None = _DENSE_VECTOR_NAME
    nearest: list[float] | rest.SparseVector | rest.FusionQuery = query.dense_vector
    prefetch = None
    if query.mode != "dense":
        if _PRIMARY_SPARSE_VECTOR_NAME not in available_sparse_vector_names:
            raise ValueError(
                "Qdrant collection does not have a BM25 index; reingest with hybrid=True"
            )
        sparse = query.sparse_vectors.get(_PRIMARY_SPARSE_VECTOR_NAME)
        if sparse is None:
            raise ValueError("BM25 query vector is required")
        sparse_query = rest.SparseVector(indices=sparse.indices, values=sparse.values)
        if query.mode == "lexical":
            using = _PRIMARY_SPARSE_VECTOR_NAME
            nearest = sparse_query
        else:
            pool = query.candidate_limit or query.limit
            prefetch = [
                rest.Prefetch(
                    query=query.dense_vector,
                    using=_DENSE_VECTOR_NAME,
                    filter=scope,
                    limit=pool,
                ),
                rest.Prefetch(
                    query=sparse_query,
                    using=_PRIMARY_SPARSE_VECTOR_NAME,
                    filter=scope,
                    limit=pool,
                ),
            ]
            nearest = rest.FusionQuery(fusion=rest.Fusion.RRF)
            using = None
    response = await client.query_points(
        collection_name=collection_name,
        query=nearest,
        using=using,
        prefetch=prefetch,
        query_filter=scope,
        limit=query.limit,
        with_payload=True,
        consistency=consistency,
    )
    return [
        candidate(_point_to_result(point, policy=policy), point.payload or {})
        for point in response.points
    ]
