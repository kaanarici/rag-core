"""TurboPuffer write-row and schema helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum

from rag_core.search.policy import VectorStorePolicy
from rag_core.search.types import SparseVector, VectorPoint

_MAX_TURBOPUFFER_ID_BYTES = 64


def _sparse_vector_to_rank_map(sparse_vector: SparseVector) -> dict[str, float]:
    return {
        f"dim{index}": value
        for index, value in zip(
            sparse_vector.indices,
            sparse_vector.values,
            strict=True,
        )
    }


def _point_to_row(point: VectorPoint) -> dict[str, object]:
    row: dict[str, object] = {
        "id": _validate_point_id(point.id),
        "vector": point.dense_vector,
    }
    sparse_row = _sparse_row_from_point(point)
    if sparse_row is not None:
        row["sparse_vector"] = sparse_row
    row.update(validate_json_payload(point.payload))
    return row


def _sparse_row_from_point(point: VectorPoint) -> dict[str, float] | None:
    sparse_vectors = point.all_sparse_vectors()
    for sparse_vector in sparse_vectors.values():
        if sparse_vector.indices:
            return _sparse_vector_to_rank_map(sparse_vector)
    return None


def validate_json_payload(payload: Mapping[str, object]) -> dict[str, object]:
    return {
        _payload_key(key): _jsonish(value)
        for key, value in payload.items()
    }


def _payload_key(key: object) -> str:
    if isinstance(key, Enum):
        return _payload_key(key.value)
    if isinstance(key, str):
        return key
    if isinstance(key, (int, float)):
        return str(key)
    raise ValueError(
        "vector payload contains unsupported key type: "
        f"{type(key).__name__}"
    )


def _jsonish(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            _payload_key(key): _jsonish(nested)
            for key, nested in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonish(nested) for nested in value]
    raise ValueError(
        "vector payload contains unsupported value type: "
        f"{type(value).__name__}"
    )


def _schema(
    dense_dimensions: int,
    *,
    policy: VectorStorePolicy,
) -> dict[str, dict[str, object]]:
    return {
        "vector": {"type": f"[{dense_dimensions}]f32", "ann": True},
        "sparse_vector": {"type": "sparse", "ann": True},
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
