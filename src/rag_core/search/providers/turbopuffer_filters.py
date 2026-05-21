"""TurboPuffer filter translation helpers."""

from __future__ import annotations

from collections.abc import Sequence

from rag_core.search.policy import VectorStorePolicy
from rag_core.search.query_plan import UnsupportedQueryStage
from rag_core.search.types import (
    And,
    DeleteFilter,
    Filter,
    Geo,
    In,
    Not,
    Or,
    Range,
    SearchQuery,
    Term,
)

from .turbopuffer_payloads import _jsonish


def _search_filter(
    *,
    query: SearchQuery,
    namespace: str,
    policy: VectorStorePolicy,
) -> object:
    filters: list[object] = [
        (policy.namespace_field, "Eq", namespace),
    ]
    if query.corpus_ids:
        filters.append((policy.corpus_id_field, "In", tuple(query.corpus_ids)))
    if query.content_types:
        filters.append((policy.content_type_field, "In", tuple(query.content_types)))
    if query.document_ids:
        filters.append((policy.document_id_field, "In", tuple(query.document_ids)))
    if query.metadata_filter is not None:
        filters.append(_metadata_filter(query.metadata_filter))
    return _combine_filters(filters)


def _delete_filter(
    *,
    filter_values: DeleteFilter,
    namespace: str,
    policy: VectorStorePolicy,
) -> object:
    filters: list[object] = [(policy.namespace_field, "Eq", namespace)]
    if filter_values.corpus_id:
        filters.append((policy.corpus_id_field, "Eq", filter_values.corpus_id))
    if filter_values.document_id:
        filters.append((policy.document_id_field, "Eq", filter_values.document_id))
    return _combine_filters(filters)


def _document_lookup_filter(
    *,
    namespace: str,
    corpus_id: str,
    document_id: str | None,
    document_key: str | None,
    policy: VectorStorePolicy,
) -> object:
    filters: list[object] = [
        (policy.namespace_field, "Eq", namespace),
        (policy.corpus_id_field, "Eq", corpus_id),
    ]
    if document_id is not None:
        filters.append((policy.document_id_field, "Eq", document_id))
    if document_key is not None:
        filters.append((policy.document_key_field, "Eq", document_key))
    return _combine_filters(filters)


def _metadata_filter(filter: Filter) -> object:
    if isinstance(filter, Term):
        return (filter.field, "Eq", _jsonish(filter.value))
    if isinstance(filter, In):
        return (filter.field, "In", tuple(_jsonish(value) for value in filter.values))
    if isinstance(filter, Range):
        filters: list[object] = []
        if filter.gte is not None:
            filters.append((filter.field, "Gte", filter.gte))
        if filter.gt is not None:
            filters.append((filter.field, "Gt", filter.gt))
        if filter.lte is not None:
            filters.append((filter.field, "Lte", filter.lte))
        if filter.lt is not None:
            filters.append((filter.field, "Lt", filter.lt))
        return _combine_filters(filters)
    if isinstance(filter, And):
        return ("And", tuple(_metadata_filter(child) for child in filter.filters))
    if isinstance(filter, Or):
        return ("Or", tuple(_metadata_filter(child) for child in filter.filters))
    if isinstance(filter, Not):
        return ("Not", _metadata_filter(filter.filter))
    if isinstance(filter, Geo):
        raise UnsupportedQueryStage("turbopuffer adapter cannot translate Geo filters")
    raise UnsupportedQueryStage(
        f"turbopuffer adapter cannot translate Filter node {type(filter).__name__}"
    )


def _combine_filters(filters: Sequence[object]) -> object:
    if len(filters) == 1:
        return filters[0]
    return ("And", tuple(filters))
