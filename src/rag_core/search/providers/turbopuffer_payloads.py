"""TurboPuffer payload, row, and filter translation helpers."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import NoReturn, Protocol, cast

from rag_core.search.filter_visitor import FilterTranslator
from rag_core.search.filters import (
    Filter,
    Geo,
    In,
    Range,
    Term,
)
from rag_core.search.policy import VectorStorePolicy
from rag_core.search.query_plan import UnsupportedQueryStage
from rag_core.search.request_models import DeleteFilter, SearchQuery
from rag_core.search.stored_payload import (
    json_payload_value,
    payload_to_result,
    validate_json_payload as _validate_json_payload,
)
from rag_core.search.vector_models import SearchResult, VectorPoint

_MAX_TURBOPUFFER_ID_BYTES = 64
TURBOPUFFER_BM25_TEXT_FIELD = "bm25_text"
_MISSING = object()


def _point_to_row(point: VectorPoint, *, policy: VectorStorePolicy) -> dict[str, object]:
    row: dict[str, object] = {
        "id": _validate_point_id(point.id),
        "vector": point.dense_vector,
    }
    payload = _validate_json_payload(point.payload)
    if TURBOPUFFER_BM25_TEXT_FIELD in payload:
        raise ValueError(
            f"payload field {TURBOPUFFER_BM25_TEXT_FIELD!r} is reserved by the "
            "TurboPuffer adapter for rank-only BM25 text; rename the metadata field"
        )
    row.update(payload)
    row[TURBOPUFFER_BM25_TEXT_FIELD] = point.sparse_text or str(
        row.get(policy.text_field) or ""
    )
    return row


def _schema(
    dense_dimensions: int,
    *,
    policy: VectorStorePolicy,
) -> dict[str, dict[str, object]]:
    return {
        "vector": {"type": f"[{dense_dimensions}]f32", "ann": True},
        policy.namespace_field: {"type": "string", "filterable": True},
        policy.collection_field: {"type": "string", "filterable": True},
        policy.document_id_field: {"type": "string", "filterable": True},
        policy.document_key_field: {"type": "string", "filterable": True},
        policy.content_sha256_field: {"type": "string", "filterable": True},
        policy.processing_version_field: {"type": "string", "filterable": True},
        policy.content_type_field: {"type": "string", "filterable": True},
        policy.source_type_field: {"type": "string", "filterable": True},
        policy.chunk_index_field: {"type": "int", "filterable": True},
        policy.text_field: {
            "type": "string",
            "filterable": False,
            "full_text_search": False,
        },
        TURBOPUFFER_BM25_TEXT_FIELD: {
            "type": "string",
            "filterable": False,
            "full_text_search": True,
        },
        policy.title_field: {"type": "string", "filterable": False},
    }


def _validate_point_id(value: object) -> str:
    point_id = _non_empty_string(
        value,
        "turbopuffer point id must be a non-empty string",
    )
    if len(point_id.encode("utf-8")) > _MAX_TURBOPUFFER_ID_BYTES:
        raise ValueError("turbopuffer point id must be at most 64 UTF-8 bytes")
    return point_id


class _ToDict(Protocol):
    def to_dict(self) -> Mapping[str, object]: ...


def _row_to_result(
    row: object,
    *,
    distance_metric: str,
    policy: VectorStorePolicy,
) -> SearchResult:
    return payload_to_result(
        point_id=_required_row_id(row),
        payload=_row_payload(row),
        score=_distance_to_score(
            _required_row_float(row, "$dist"),
            distance_metric=distance_metric,
        ),
        policy=policy,
    )


def _required_response_rows(response: object, *, operation: str) -> list[object]:
    rows = getattr(response, "rows", _MISSING)
    if rows is _MISSING or rows is None:
        raise ValueError("turbopuffer %s response missing required rows" % operation)
    try:
        return list(cast(Iterable[object], rows))
    except TypeError:
        raise ValueError("turbopuffer %s response returned invalid rows" % operation) from None


def _row_payload(row: object) -> dict[str, object]:
    if isinstance(row, Mapping):
        raw = dict(row)
    elif hasattr(row, "model_extra"):
        raw = dict(getattr(row, "model_extra") or {})
        raw["id"] = getattr(row, "id")
    elif hasattr(row, "to_dict"):
        raw = dict(cast(_ToDict, row).to_dict())
    else:
        raw = dict(getattr(row, "__dict__", {}))

    return {
        str(key): value
        for key, value in raw.items()
        if key not in {"id", "vector", "$dist", TURBOPUFFER_BM25_TEXT_FIELD}
    }


def _required_row_value(row: object, key: str) -> object:
    value = _row_value(row, key, default=_MISSING)
    if value is _MISSING:
        raise ValueError("turbopuffer result row missing required field: %s" % key)
    return value


def _required_row_float(row: object, key: str) -> float:
    value = _required_row_value(row, key)
    parsed = _optional_float(value)
    if parsed is None:
        raise ValueError("turbopuffer result row returned invalid field: %s" % key)
    if key == "$dist" and parsed < 0.0:
        raise ValueError("turbopuffer result row returned invalid field: %s" % key)
    return parsed


def _required_row_id(row: object) -> str:
    value = _required_row_value(row, "id")
    return _non_empty_string(
        value,
        "turbopuffer result row missing required field: id",
    )


def _non_empty_string(value: object, message: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(message)
    return value


def _row_value(row: object, key: str, *, default: object = _MISSING) -> object:
    if isinstance(row, Mapping):
        return row.get(key, default)
    try:
        return row[key]  # type: ignore[index]
    except (IndexError, KeyError, TypeError):
        return getattr(row, key, default)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        parsed = float(value)
    except (OverflowError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _distance_to_score(
    distance: float,
    *,
    distance_metric: str,
) -> float:
    if distance_metric == "cosine_distance":
        return 1.0 - min(distance, 2.0)
    return 1.0 / (1.0 + distance)


def _search_filter(
    *,
    query: SearchQuery,
    namespace: str,
    policy: VectorStorePolicy,
) -> object:
    filters: list[object] = [
        (policy.namespace_field, "Eq", namespace),
    ]
    if query.collections:
        filters.append((policy.collection_field, "In", tuple(query.collections)))
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
    if filter_values.collection:
        filters.append((policy.collection_field, "Eq", filter_values.collection))
    if filter_values.document_id:
        filters.append((policy.document_id_field, "Eq", filter_values.document_id))
    return _combine_filters(filters)


def _document_lookup_filter(
    *,
    namespace: str,
    collection: str,
    document_id: str | None,
    document_key: str | None,
    policy: VectorStorePolicy,
) -> object:
    filters: list[object] = [
        (policy.namespace_field, "Eq", namespace),
        (policy.collection_field, "Eq", collection),
    ]
    if document_id is not None:
        filters.append((policy.document_id_field, "Eq", document_id))
    if document_key is not None:
        filters.append((policy.document_key_field, "Eq", document_key))
    return _combine_filters(filters)


def _metadata_filter(filter: Filter) -> object:
    return _TurbopufferFilterTranslator().translate(filter)


class _TurbopufferFilterTranslator(FilterTranslator[object]):
    def term(self, node: Term) -> object:
        return (node.field, "Eq", json_payload_value(node.value))

    def in_(self, node: In) -> object:
        return (
            node.field,
            "In",
            tuple(json_payload_value(value) for value in node.values),
        )

    def range_(self, node: Range) -> object:
        filters: list[object] = []
        if node.gte is not None:
            filters.append((node.field, "Gte", node.gte))
        if node.gt is not None:
            filters.append((node.field, "Gt", node.gt))
        if node.lte is not None:
            filters.append((node.field, "Lte", node.lte))
        if node.lt is not None:
            filters.append((node.field, "Lt", node.lt))
        return _combine_filters(filters)

    def geo(self, node: Geo) -> object:
        raise UnsupportedQueryStage("turbopuffer adapter cannot translate Geo filters")

    def and_(self, children: list[object]) -> object:
        return ("And", tuple(children))

    def or_(self, children: list[object]) -> object:
        return ("Or", tuple(children))

    def not_(self, child: object) -> object:
        return ("Not", child)

    def unsupported(self, node: Filter) -> NoReturn:
        raise UnsupportedQueryStage(
            f"turbopuffer adapter cannot translate Filter node {type(node).__name__}"
        )


def _combine_filters(filters: Sequence[object]) -> object:
    if len(filters) == 1:
        return filters[0]
    return ("And", tuple(filters))
