"""TurboPuffer write-row and schema helpers."""

from __future__ import annotations

from rag_core.search.policy import VectorStorePolicy
from rag_core.search.stored_payload import validate_json_payload as _validate_json_payload
from rag_core.search.vector_models import VectorPoint

_MAX_TURBOPUFFER_ID_BYTES = 64
TURBOPUFFER_BM25_TEXT_FIELD = "bm25_text"


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
        policy.corpus_id_field: {"type": "string", "filterable": True},
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


def _non_empty_string(value: object, message: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(message)
    return value
