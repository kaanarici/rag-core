from __future__ import annotations

from rag_core.search.filter_eval import eval_filter
from rag_core.search.stored_payload_fields import SEARCH_RESULT_FILTER_FIELDS
from rag_core.search.request_models import SearchSidecarQuery
from rag_core.search.vector_models import SearchResult


def search_result_payload(result: SearchResult) -> dict[str, object]:
    payload = dict(result.metadata)
    for key in SEARCH_RESULT_FILTER_FIELDS:
        value = getattr(result, key)
        if value is not None:
            payload[key] = value
    return payload


def result_matches_sidecar_query(
    result: SearchResult,
    query: SearchSidecarQuery,
    *,
    allow_missing_namespace: bool = False,
) -> bool:
    result_namespace = _result_namespace(result)
    if result_namespace is None:
        if not allow_missing_namespace:
            return False
    elif result_namespace != query.namespace:
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
    if query.metadata_filter is not None and not eval_filter(
        query.metadata_filter,
        search_result_payload(result),
    ):
        return False
    return True


def _result_namespace(result: SearchResult) -> str | None:
    if result.namespace:
        return result.namespace
    value = result.metadata.get("namespace")
    if isinstance(value, str) and value:
        return value
    return None
