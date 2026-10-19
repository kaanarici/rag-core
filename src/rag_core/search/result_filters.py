from __future__ import annotations

from rag_core.search.request_models import SearchQuery
from rag_core.search.vector_models import SearchResult


def result_matches_query(
    result: SearchResult,
    query: SearchQuery,
    *,
    allow_missing_tenant_id: bool = False,
) -> bool:
    """Check returned scope after the backend has applied metadata filters."""
    if query.metadata_filter is not None:
        raise ValueError("metadata filters must be applied by the vector store")
    result_tenant_id = _result_tenant_id(result)
    if result_tenant_id is None:
        if not allow_missing_tenant_id:
            return False
    elif result_tenant_id != query.tenant_id:
        return False
    if query.collections is not None and not query.collections:
        return False
    if query.collections and result.collection not in query.collections:
        return False
    if query.content_types is not None and not query.content_types:
        return False
    if query.content_types and result.content_type not in query.content_types:
        return False
    if query.document_ids is not None and not query.document_ids:
        return False
    if query.document_ids and result.document_id not in query.document_ids:
        return False
    return True


def _result_tenant_id(result: SearchResult) -> str | None:
    if result.tenant_id:
        return result.tenant_id
    value = result.metadata.get("tenant_id")
    if isinstance(value, str) and value:
        return value
    return None
