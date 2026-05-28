"""Qdrant point, result, and document-record conversion helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import Enum
from typing import Any, Mapping
from uuid import UUID

from qdrant_client import models as rest

from rag_core.search.policy import VectorStorePolicy
from rag_core.search.request_models import StoredDocumentRecord
from rag_core.search.stored_payload import payload_to_result
from rag_core.search.vector_models import SearchResult, VectorPoint

from .qdrant_shared import _DENSE_VECTOR_NAME, _KNOWN_SPARSE_VECTOR_NAMES


def validate_json_payload(payload: Mapping[str, object]) -> dict[str, object]:
    return {_payload_key(key): _jsonish(value) for key, value in payload.items()}


def _payload_key(key: object) -> str:
    if isinstance(key, Enum):
        return _payload_key(key.value)
    if isinstance(key, str):
        return key
    if isinstance(key, (int, float)):
        return str(key)
    raise ValueError(
        f"vector payload contains unsupported key type: {type(key).__name__}"
    )


def _jsonish(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {_payload_key(key): _jsonish(nested) for key, nested in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonish(nested) for nested in value]
    raise ValueError(
        f"vector payload contains unsupported value type: {type(value).__name__}"
    )

_MISSING = object()


def _build_point(
    point: VectorPoint,
    available_sparse_vector_names: frozenset[str]
    | set[str] = _KNOWN_SPARSE_VECTOR_NAMES,
) -> rest.PointStruct:
    point_id = _qdrant_point_id(point.id)
    all_sparse_vectors = point.all_sparse_vectors()
    missing_sparse_vectors = sorted(
        name
        for name, vector in all_sparse_vectors.items()
        if name in _KNOWN_SPARSE_VECTOR_NAMES
        and name not in available_sparse_vector_names
        and (vector.indices or vector.values)
    )
    if missing_sparse_vectors:
        available = ", ".join(sorted(available_sparse_vector_names)) or "none"
        missing = ", ".join(missing_sparse_vectors)
        raise ValueError(
            "Qdrant collection is missing sparse vector channels required by "
            f"points: {missing}; available sparse channels: {available}"
        )
    sparse_dict = {
        name: rest.SparseVector(indices=vector.indices, values=vector.values)
        for name, vector in all_sparse_vectors.items()
        if name in available_sparse_vector_names
    }
    vector: dict[str, Any] = {
        _DENSE_VECTOR_NAME: point.dense_vector,
        **sparse_dict,
    }
    return rest.PointStruct(
        id=point_id,
        vector=vector,
        payload=validate_json_payload(point.payload),
    )


def _point_to_result(
    point: rest.ScoredPoint, *, policy: VectorStorePolicy
) -> SearchResult:
    return payload_to_result(
        point_id=_required_point_id(point),
        payload=point.payload or {},
        score=_score_result_value(getattr(point, "score", _MISSING)),
        policy=policy,
    )


def _required_point_id(point: rest.ScoredPoint) -> str:
    value = getattr(point, "id", _MISSING)
    if (
        value is _MISSING
        or value is None
        or isinstance(value, bool)
        or not isinstance(value, (str, UUID))
        or (isinstance(value, str) and not value.strip())
    ):
        raise ValueError("qdrant result point missing required field: id")
    try:
        return _qdrant_point_id(value)
    except ValueError as exc:
        raise ValueError("qdrant result point missing required field: id") from exc


def _qdrant_point_id(value: str | UUID) -> str:
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Qdrant point IDs must be UUID strings. Use the default "
            "VectorStorePolicy point_id_format or choose a non-Qdrant store "
            "for custom point IDs."
        ) from exc


def _score_result_value(value: object) -> float:
    if value is _MISSING or value is None:
        raise ValueError("qdrant result point missing required field: score")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("qdrant result point returned invalid field: score")
    try:
        parsed = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(
            "qdrant result point returned invalid field: score"
        ) from exc
    if not math.isfinite(parsed):
        raise ValueError("qdrant result point returned invalid field: score")
    return parsed


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


def _resolve_document_id(
    *,
    payload: Mapping[str, object],
    fallback_document_id: str | None,
    policy: VectorStorePolicy,
) -> str:
    document_id = _payload_string(payload, policy.document_id_field)
    if document_id is not None:
        return document_id
    return fallback_document_id or ""


def _count_result_value(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("qdrant document count response missing valid count")
    return value


def _build_stored_document_record(
    *,
    payload: Mapping[str, object],
    namespace: str,
    corpus_id: str,
    document_id: str,
    chunk_count: int,
    policy: VectorStorePolicy,
) -> StoredDocumentRecord:
    return StoredDocumentRecord(
        document_id=document_id,
        namespace=namespace,
        corpus_id=corpus_id,
        document_key=_payload_optional_str(payload, policy.document_key_field),
        content_sha256=_payload_optional_str(payload, policy.content_sha256_field),
        processing_version=_payload_optional_str(
            payload, policy.processing_version_field
        ),
        chunk_count=chunk_count,
    )


def _payload_optional_str(payload: Mapping[str, object], key: str) -> str | None:
    return _payload_string(payload, key)


def _payload_string(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"qdrant document record payload field {key!r} must be a string"
        )
    return value
