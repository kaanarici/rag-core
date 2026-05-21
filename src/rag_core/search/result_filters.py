from __future__ import annotations

from rag_core.search.filter_eval import eval_filter
from rag_core.search.types import SearchResult, SearchSidecarQuery

_RESULT_FILTER_FIELDS = (
    "namespace",
    "content_type",
    "source_type",
    "document_id",
    "corpus_id",
    "document_key",
    "content_sha256",
    "title",
    "section_id",
    "section_title",
    "section_path",
    "document_path",
    "chunk_index",
    "chunk_word_count",
    "chunk_token_estimate",
    "embedding_model",
    "chunker_strategy",
    "result_type",
    "figure_id",
    "figure_thumbnail_url",
)


def search_result_payload(result: SearchResult) -> dict[str, object]:
    payload = dict(result.metadata)
    for key in _RESULT_FILTER_FIELDS:
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
    if query.corpus_ids is not None and not query.corpus_ids:
        return False
    if query.corpus_ids and result.corpus_id not in query.corpus_ids:
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
