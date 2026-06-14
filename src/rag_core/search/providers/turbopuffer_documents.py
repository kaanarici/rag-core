"""TurboPuffer document-record lookup helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping

from rag_core.search.document_records import (
    resolve_document_id_from_payload,
    stored_document_record_from_payload,
    validate_document_lookup_inputs,
)
from rag_core.search.policy import VectorStorePolicy
from rag_core.search.request_models import StoredDocumentRecord

from .turbopuffer_client import TurboPufferNamespace
from .turbopuffer_filters import _document_lookup_filter
from .turbopuffer_rows import _required_response_rows, _row_payload


async def get_turbopuffer_document_record(
    *,
    namespace_client: TurboPufferNamespace,
    namespace: str,
    corpus_id: str,
    document_id: str | None,
    document_key: str | None,
    policy: VectorStorePolicy,
) -> StoredDocumentRecord | None:
    namespace_scoped, corpus_scoped = validate_document_lookup_inputs(
        namespace=namespace,
        corpus_id=corpus_id,
        document_id=document_id,
        document_key=document_key,
    )
    filters = _document_lookup_filter(
        namespace=namespace_scoped,
        corpus_id=corpus_scoped,
        document_id=document_id,
        document_key=document_key,
        policy=policy,
    )
    lookup_response = await namespace_client.query(
        rank_by=("id", "asc"),
        filters=filters,
        limit=1,
        include_attributes=True,
    )
    rows = _required_response_rows(lookup_response, operation="document lookup")
    if not rows:
        return None
    sample = _row_payload(rows[0])
    resolved_document_id = resolve_document_id_from_payload(
        payload=sample,
        document_id_field=policy.document_id_field,
        fallback_document_id=document_id,
        invalid_message=(
            "turbopuffer document lookup returned invalid string field: "
            f"{policy.document_id_field}"
        ),
        reject_blank=True,
    )
    if not resolved_document_id:
        return None
    count_response = await namespace_client.query(
        filters=_document_lookup_filter(
            namespace=namespace_scoped,
            corpus_id=corpus_scoped,
            document_id=resolved_document_id,
            document_key=None,
            policy=policy,
        ),
        limit=1,
        aggregate_by={"chunk_count": ("Count",)},
    )

    return stored_document_record_from_payload(
        payload=sample,
        namespace=namespace_scoped,
        corpus_id=corpus_scoped,
        document_id=resolved_document_id,
        chunk_count=_response_aggregation_int(count_response, "chunk_count"),
        policy=policy,
        invalid_field_message=(
            "turbopuffer document lookup returned invalid string field: {field}"
        ),
    )


def _response_aggregation_int(response: object, key: str) -> int:
    aggregations = getattr(response, "aggregations", None)
    if not isinstance(aggregations, Mapping):
        raise ValueError(
            "turbopuffer document lookup missing required aggregation: %s" % key
        )
    if key not in aggregations:
        raise ValueError(
            "turbopuffer document lookup missing required aggregation: %s" % key
        )
    value = aggregations.get(key)
    parsed = _parse_non_negative_int(value)
    if parsed is None:
        raise ValueError(
            "turbopuffer document lookup returned invalid aggregation: %s" % key
        )
    return parsed


def _parse_non_negative_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None
        parsed = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped.isdecimal():
            return None
        parsed = int(stripped)
    else:
        return None
    if parsed < 0:
        return None
    return parsed
