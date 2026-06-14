"""Qdrant point, result, and document-record conversion helpers."""

from __future__ import annotations

import math
from typing import Any
from uuid import UUID

from qdrant_client import models as rest

from rag_core.search.policy import VectorStorePolicy
from rag_core.search.stored_payload import payload_to_result
from rag_core.search.stored_payload import validate_json_payload as _validate_json_payload
from rag_core.search.vector_models import SearchResult, VectorPoint

from .qdrant_shared import _DENSE_VECTOR_NAME, _KNOWN_SPARSE_VECTOR_NAMES


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
        payload=_validate_json_payload(point.payload),
    )


def _point_to_result(
    point: rest.ScoredPoint, *, policy: VectorStorePolicy
) -> SearchResult:
    return payload_to_result(
        point_id=_required_point_id(point, source="qdrant result point"),
        payload=point.payload or {},
        score=_score_result_value(getattr(point, "score", _MISSING)),
        policy=policy,
    )


def _record_to_result(record: rest.Record, *, policy: VectorStorePolicy) -> SearchResult:
    return payload_to_result(
        point_id=_required_point_id(record, source="qdrant record"),
        payload=record.payload or {},
        score=0.0,
        policy=policy,
    )


def _required_point_id(point: object, *, source: str) -> str:
    value = getattr(point, "id", _MISSING)
    if (
        value is _MISSING
        or value is None
        or isinstance(value, bool)
        or not isinstance(value, (str, UUID))
        or (isinstance(value, str) and not value.strip())
    ):
        raise ValueError(f"{source} missing required field: id")
    try:
        return _qdrant_point_id(value)
    except ValueError as exc:
        raise ValueError(f"{source} missing required field: id") from exc


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


def _count_result_value(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("qdrant document count response missing valid count")
    return value
