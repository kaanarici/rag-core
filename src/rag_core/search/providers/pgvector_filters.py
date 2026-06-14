from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from rag_core.search.filters import And, Filter, Geo, In, Not, Or, Range, Term
from rag_core.search.policy import VectorStorePolicy
from rag_core.search.request_models import DeleteFilter, SearchQuery
from rag_core.search.stored_payload import json_payload_value

from .pgvector_config import quote_identifier

_PAYLOAD_FIELD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]{0,127}$")
_SQL_FLOAT_TEXT_RE = r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$"


@dataclass(frozen=True)
class SqlFragment:
    sql: str
    params: tuple[object, ...]


class _SqlBuilder:
    def __init__(self, *, start_index: int = 1) -> None:
        self._start_index = start_index
        self.params: list[object] = []

    def add_param(self, value: object) -> str:
        self.params.append(value)
        return f"${self._start_index + len(self.params) - 1}"

    def fragment(self, sql: str) -> SqlFragment:
        return SqlFragment(sql=sql, params=tuple(self.params))


def build_search_where(
    *,
    query: SearchQuery,
    namespace: str,
    policy: VectorStorePolicy,
    start_index: int,
) -> SqlFragment:
    builder = _SqlBuilder(start_index=start_index)
    filters = [_equals(policy.namespace_field, namespace, policy=policy, builder=builder)]
    if query.corpus_ids:
        filters.append(
            _in_values(
                policy.corpus_id_field,
                tuple(query.corpus_ids),
                policy=policy,
                builder=builder,
            )
        )
    if query.content_types:
        filters.append(
            _in_values(
                policy.content_type_field,
                tuple(query.content_types),
                policy=policy,
                builder=builder,
            )
        )
    if query.document_ids:
        filters.append(
            _in_values(
                policy.document_id_field,
                tuple(query.document_ids),
                policy=policy,
                builder=builder,
            )
        )
    if query.metadata_filter is not None:
        filters.append(_metadata_filter(query.metadata_filter, policy=policy, builder=builder))
    return builder.fragment(_combine_sql("AND", filters))


def build_delete_where(
    *,
    filter_values: DeleteFilter,
    namespace: str,
    policy: VectorStorePolicy,
    start_index: int = 1,
) -> SqlFragment:
    builder = _SqlBuilder(start_index=start_index)
    filters = [_equals(policy.namespace_field, namespace, policy=policy, builder=builder)]
    if filter_values.corpus_id is not None:
        filters.append(
            _equals(
                policy.corpus_id_field,
                filter_values.corpus_id,
                policy=policy,
                builder=builder,
            )
        )
    if filter_values.document_id is not None:
        filters.append(
            _equals(
                policy.document_id_field,
                filter_values.document_id,
                policy=policy,
                builder=builder,
            )
        )
    return builder.fragment(_combine_sql("AND", filters))


def build_document_lookup_where(
    *,
    namespace: str,
    corpus_id: str,
    document_id: str | None,
    document_key: str | None,
    policy: VectorStorePolicy,
    start_index: int = 1,
) -> SqlFragment:
    builder = _SqlBuilder(start_index=start_index)
    filters = [
        _equals(policy.namespace_field, namespace, policy=policy, builder=builder),
        _equals(policy.corpus_id_field, corpus_id, policy=policy, builder=builder),
    ]
    if document_id is not None:
        filters.append(
            _equals(policy.document_id_field, document_id, policy=policy, builder=builder)
        )
    if document_key is not None:
        filters.append(
            _equals(policy.document_key_field, document_key, policy=policy, builder=builder)
        )
    return builder.fragment(_combine_sql("AND", filters))


def build_chunk_lookup_where(
    *,
    namespace: str,
    corpus_id: str,
    document_id: str,
    chunk_indices: tuple[int, ...],
    policy: VectorStorePolicy,
    start_index: int = 1,
) -> SqlFragment:
    builder = _SqlBuilder(start_index=start_index)
    filters = [
        _equals(policy.namespace_field, namespace, policy=policy, builder=builder),
        _equals(policy.corpus_id_field, corpus_id, policy=policy, builder=builder),
        _equals(policy.document_id_field, document_id, policy=policy, builder=builder),
        _in_values(
            policy.chunk_index_field,
            chunk_indices,
            policy=policy,
            builder=builder,
        ),
    ]
    return builder.fragment(_combine_sql("AND", filters))


def _metadata_filter(
    filter_node: Filter,
    *,
    policy: VectorStorePolicy,
    builder: _SqlBuilder,
) -> str:
    if isinstance(filter_node, Term):
        return _equals(filter_node.field, filter_node.value, policy=policy, builder=builder)
    if isinstance(filter_node, In):
        return _in_values(
            filter_node.field,
            filter_node.values,
            policy=policy,
            builder=builder,
        )
    if isinstance(filter_node, Range):
        return _range(filter_node, policy=policy, builder=builder)
    if isinstance(filter_node, Geo):
        return _geo(filter_node, builder=builder)
    if isinstance(filter_node, And):
        return _combine_sql(
            "AND",
            [
                _metadata_filter(child, policy=policy, builder=builder)
                for child in filter_node.filters
            ],
        )
    if isinstance(filter_node, Or):
        return _combine_sql(
            "OR",
            [
                _metadata_filter(child, policy=policy, builder=builder)
                for child in filter_node.filters
            ],
        )
    if isinstance(filter_node, Not):
        return f"(NOT {_metadata_filter(filter_node.filter, policy=policy, builder=builder)})"
    raise TypeError(f"unsupported pgvector filter node: {type(filter_node).__name__}")


def _equals(
    field: str,
    value: object,
    *,
    policy: VectorStorePolicy,
    builder: _SqlBuilder,
) -> str:
    field_ref = _field_ref(field, policy=policy, builder=builder)
    if field_ref.kind == "integer_column":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"pgvector filter field {field!r} requires an integer")
        return f"{field_ref.sql} = {builder.add_param(value)}::integer"
    if field_ref.kind == "text_column":
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"pgvector filter field {field!r} requires a scalar value")
        return f"{field_ref.sql} = {builder.add_param(str(value))}"
    return f"{field_ref.sql} = {builder.add_param(_jsonb_param(value))}::jsonb"


def _in_values(
    field: str,
    values: tuple[object, ...],
    *,
    policy: VectorStorePolicy,
    builder: _SqlBuilder,
) -> str:
    if not values:
        raise ValueError("pgvector In filter values must not be empty")
    field_ref = _field_ref(field, policy=policy, builder=builder)
    if field_ref.kind == "integer_column":
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError(f"pgvector filter field {field!r} requires integer values")
        return f"{field_ref.sql} = ANY({builder.add_param(list(values))}::integer[])"
    if field_ref.kind == "text_column":
        return (
            f"{field_ref.sql} = ANY("
            f"{builder.add_param([str(value) for value in values])}::text[])"
        )
    comparisons = [
        f"{field_ref.sql} = {builder.add_param(_jsonb_param(value))}::jsonb"
        for value in values
    ]
    return _combine_sql("OR", comparisons)


def _range(
    filter_node: Range,
    *,
    policy: VectorStorePolicy,
    builder: _SqlBuilder,
) -> str:
    field_ref = _field_ref(filter_node.field, policy=policy, builder=builder)
    kind = _range_kind(filter_node)
    bounds = (
        (">=", filter_node.gte),
        (">", filter_node.gt),
        ("<=", filter_node.lte),
        ("<", filter_node.lt),
    )
    comparisons: list[str] = []
    for operator, value in bounds:
        if value is None:
            continue
        if field_ref.kind == "integer_column":
            comparisons.append(
                _coalesced_comparison(
                    field_ref.sql,
                    operator,
                    builder.add_param(value),
                )
            )
        elif kind == "numeric":
            comparisons.append(
                _coalesced_comparison(
                    _typed_json_value_sql(field_ref, kind="numeric"),
                    operator,
                    builder.add_param(float(value)),
                )
            )
        else:
            comparisons.append(
                _coalesced_comparison(
                    _typed_json_value_sql(field_ref, kind="string"),
                    operator,
                    builder.add_param(value),
                )
            )
    return _combine_sql("AND", comparisons)


def _geo(filter_node: Geo, *, builder: _SqlBuilder) -> str:
    field_name = _validate_payload_field(filter_node.field)
    field_param = builder.add_param(field_name)
    lat_param = builder.add_param(filter_node.lat)
    lon_param = builder.add_param(filter_node.lon)
    radius_param = builder.add_param(filter_node.radius_m)
    point_sql = f"(payload -> {field_param})"
    lat_sql = _safe_float_text_sql(f"({point_sql} ->> 'lat')")
    lon_sql = _safe_float_text_sql(f"({point_sql} ->> 'lon')")
    distance_sql = (
        "6371000.0 * 2.0 * asin(least(1.0, sqrt("
        f"power(sin(radians(({lat_param} - {lat_sql}) / 2.0)), 2) + "
        f"cos(radians({lat_param})) * cos(radians({lat_sql})) * "
        f"power(sin(radians(({lon_param} - {lon_sql}) / 2.0)), 2)"
        ")))"
    )
    return f"COALESCE(({distance_sql} <= {radius_param}), false)"


def _typed_json_value_sql(
    field_ref: _FieldRef,
    *,
    kind: Literal["numeric", "string"],
) -> str:
    if field_ref.kind != "jsonb":
        return field_ref.text_sql
    if kind == "numeric":
        return (
            "(CASE WHEN "
            f"jsonb_typeof({field_ref.sql}) = 'number' "
            f"THEN ({field_ref.text_sql})::double precision END)"
        )
    return (
        "(CASE WHEN "
        f"jsonb_typeof({field_ref.sql}) = 'string' "
        f"THEN {field_ref.text_sql} END)"
    )


def _safe_float_text_sql(text_sql: str) -> str:
    return (
        "(CASE WHEN "
        f"{text_sql} ~ '{_SQL_FLOAT_TEXT_RE}' "
        f"THEN ({text_sql})::double precision END)"
    )


def _coalesced_comparison(left_sql: str, operator: str, right_sql: str) -> str:
    return f"COALESCE(({left_sql} {operator} {right_sql}), false)"


@dataclass(frozen=True)
class _FieldRef:
    sql: str
    text_sql: str
    kind: Literal["jsonb", "integer_column", "text_column"]


def _field_ref(
    field: str,
    *,
    policy: VectorStorePolicy,
    builder: _SqlBuilder,
) -> _FieldRef:
    column = _promoted_column(field, policy=policy)
    if column is not None:
        column_sql = quote_identifier(column)
        kind: Literal["integer_column", "text_column"] = (
            "integer_column" if column == "chunk_index" else "text_column"
        )
        return _FieldRef(sql=column_sql, text_sql=column_sql, kind=kind)
    field_param = builder.add_param(_validate_payload_field(field))
    return _FieldRef(
        sql=f"(payload -> {field_param})",
        text_sql=f"(payload ->> {field_param})",
        kind="jsonb",
    )


def _promoted_column(field: str, *, policy: VectorStorePolicy) -> str | None:
    columns = {
        policy.namespace_field: "namespace",
        policy.corpus_id_field: "corpus_id",
        policy.document_id_field: "document_id",
        policy.document_key_field: "document_key",
        policy.content_sha256_field: "content_sha256",
        policy.processing_version_field: "processing_version",
        policy.content_type_field: "content_type",
        policy.source_type_field: "source_type",
        policy.chunk_index_field: "chunk_index",
    }
    return columns.get(field)


def _validate_payload_field(field: object) -> str:
    if not isinstance(field, str) or not field.strip():
        raise ValueError("pgvector metadata filter field must be a non-empty string")
    normalized = field.strip()
    if normalized != field or not _PAYLOAD_FIELD_RE.fullmatch(normalized):
        raise ValueError(
            "pgvector metadata filter field must match "
            "[A-Za-z_][A-Za-z0-9_.:-]{0,127}"
        )
    return normalized


def _jsonb_param(value: object) -> str:
    return json.dumps(json_payload_value(value), separators=(",", ":"))


def _range_kind(filter_node: Range) -> Literal["numeric", "string"]:
    values = [
        value
        for value in (filter_node.gte, filter_node.gt, filter_node.lte, filter_node.lt)
        if value is not None
    ]
    if all(isinstance(value, str) for value in values):
        return "string"
    if all(not isinstance(value, bool) and isinstance(value, (int, float)) for value in values):
        return "numeric"
    raise ValueError("pgvector Range filter bounds must be all numeric or all strings")


def _combine_sql(operator: Literal["AND", "OR"], expressions: list[str]) -> str:
    if not expressions:
        raise ValueError("pgvector filter expression list must not be empty")
    if len(expressions) == 1:
        return expressions[0]
    return "(" + f" {operator} ".join(expressions) + ")"
