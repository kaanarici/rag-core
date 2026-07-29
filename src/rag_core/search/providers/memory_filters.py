"""In-memory vector-store filtering helpers."""

from __future__ import annotations

from rag_core.search.filter_eval import eval_filter
from rag_core.search.policy import VectorStorePolicy
from rag_core.search.request_models import DeleteFilter, SearchQuery
from rag_core.search.stored_payload_fields import optional_payload_str

from .memory_query_scoring import MemoryPoint


def matches_memory_search_filter(
    stored: MemoryPoint,
    query: SearchQuery,
    *,
    namespace: str,
    policy: VectorStorePolicy,
) -> bool:
    payload = stored.payload
    if optional_payload_str(payload, policy.namespace_field) != namespace:
        return False
    if query.collections is not None and not query.collections:
        return False
    if (
        query.collections
        and optional_payload_str(payload, policy.collection_field) not in query.collections
    ):
        return False
    if query.content_types is not None and not query.content_types:
        return False
    if (
        query.content_types
        and optional_payload_str(payload, policy.content_type_field)
        not in query.content_types
    ):
        return False
    if query.document_ids is not None and not query.document_ids:
        return False
    if (
        query.document_ids
        and optional_payload_str(payload, policy.document_id_field)
        not in query.document_ids
    ):
        return False
    if query.metadata_filter is not None and not eval_filter(
        query.metadata_filter, payload
    ):
        return False
    return True


def matches_memory_delete_filter(
    stored: MemoryPoint,
    filter_values: DeleteFilter,
    *,
    namespace: str,
    policy: VectorStorePolicy,
) -> bool:
    payload = stored.payload
    if optional_payload_str(payload, policy.namespace_field) != namespace:
        return False
    if (
        filter_values.collection
        and optional_payload_str(payload, policy.collection_field)
        != filter_values.collection
    ):
        return False
    if (
        filter_values.document_id
        and optional_payload_str(payload, policy.document_id_field)
        != filter_values.document_id
    ):
        return False
    return True
