"""In-memory vector-store filtering helpers."""

from __future__ import annotations

from rag_core.search.filter_eval import eval_filter
from rag_core.search.policy import VectorStorePolicy
from rag_core.search.types import DeleteFilter, SearchQuery

from .memory_query_scoring import MemoryPoint


def matches_memory_search_filter(
    stored: MemoryPoint,
    query: SearchQuery,
    *,
    namespace: str,
    policy: VectorStorePolicy,
) -> bool:
    payload = stored.payload
    if str(payload.get(policy.namespace_field) or "") != namespace:
        return False
    if query.corpus_ids is not None and not query.corpus_ids:
        return False
    if (
        query.corpus_ids
        and str(payload.get(policy.corpus_id_field) or "") not in query.corpus_ids
    ):
        return False
    if query.content_types is not None and not query.content_types:
        return False
    if (
        query.content_types
        and str(payload.get(policy.content_type_field) or "") not in query.content_types
    ):
        return False
    if query.document_ids is not None and not query.document_ids:
        return False
    if (
        query.document_ids
        and str(payload.get(policy.document_id_field) or "") not in query.document_ids
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
    if str(payload.get(policy.namespace_field) or "") != namespace:
        return False
    if (
        filter_values.corpus_id
        and str(payload.get(policy.corpus_id_field) or "") != filter_values.corpus_id
    ):
        return False
    if (
        filter_values.document_id
        and str(payload.get(policy.document_id_field) or "")
        != filter_values.document_id
    ):
        return False
    return True
