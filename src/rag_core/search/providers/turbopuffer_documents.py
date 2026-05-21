"""TurboPuffer document-record lookup helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping

from rag_core.search.policy import VectorStorePolicy
from rag_core.search.types import StoredDocumentRecord

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
    namespace_scoped, corpus_scoped = _validate_document_lookup_inputs(
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
    resolved_document_id = _payload_document_id(
        sample,
        policy.document_id_field,
        fallback=document_id,
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

    return _stored_document_record_from_lookup(
        count_response=count_response,
        sample=sample,
        namespace=namespace_scoped,
        corpus_id=corpus_scoped,
        document_id=resolved_document_id,
        policy=policy,
    )


def _validate_document_lookup_inputs(
    *,
    namespace: str,
    corpus_id: str,
    document_id: str | None,
    document_key: str | None,
) -> tuple[str, str]:
    namespace_scoped = namespace.strip()
    if not namespace_scoped:
        raise ValueError("namespace is required for get_document_record")
    corpus_scoped = corpus_id.strip()
    if not corpus_scoped:
        raise ValueError("corpus_id is required for get_document_record")
    if document_id is None and document_key is None:
        raise ValueError(
            "document_id or document_key is required for get_document_record"
        )
    return namespace_scoped, corpus_scoped


def _stored_document_record_from_lookup(
    *,
    count_response: object,
    sample: Mapping[str, object],
    namespace: str,
    corpus_id: str,
    document_id: str,
    policy: VectorStorePolicy,
) -> StoredDocumentRecord:
    return StoredDocumentRecord(
        document_id=document_id,
        namespace=namespace,
        corpus_id=corpus_id,
        document_key=_payload_optional_str(sample, policy.document_key_field),
        content_sha256=_payload_optional_str(sample, policy.content_sha256_field),
        processing_version=_payload_optional_str(
            sample, policy.processing_version_field
        ),
        chunk_count=_response_aggregation_int(count_response, "chunk_count"),
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


def _payload_optional_str(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            "turbopuffer document lookup returned invalid string field: %s" % key
        )
    return value


def _payload_document_id(
    payload: Mapping[str, object],
    key: str,
    *,
    fallback: str | None,
) -> str | None:
    value = payload.get(key)
    if value is None:
        value = fallback
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "turbopuffer document lookup returned invalid string field: %s" % key
        )
    return value
