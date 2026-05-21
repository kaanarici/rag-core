from __future__ import annotations

from rag_core.search.types import QueryPlanCapabilities

MEMORY_QUERY_PLAN_CAPABILITIES = QueryPlanCapabilities(
    dense=True,
    sparse=True,
    hybrid_rrf=True,
)

QDRANT_QUERY_PLAN_CAPABILITIES = QueryPlanCapabilities(
    dense=True,
    sparse=True,
    hybrid_rrf=True,
    hybrid_dbsf=True,
    hybrid_weighted_rrf=True,
    mmr=True,
    boost=True,
    nested_prefetch=True,
)

TURBOPUFFER_QUERY_PLAN_CAPABILITIES = QueryPlanCapabilities(
    dense=True,
    sparse=True,
    hybrid_rrf=True,
)
