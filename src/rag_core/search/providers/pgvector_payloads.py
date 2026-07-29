from __future__ import annotations

import json
from collections.abc import Mapping

from rag_core.search.policy import VectorStorePolicy
from rag_core.search.stored_payload import payload_to_result
from rag_core.search.stored_payload import validate_json_payload as _validate_json_payload
from rag_core.search.stored_payload_fields import optional_payload_int, optional_payload_str
from rag_core.search.vector_models import SearchResult, VectorPoint


def point_to_pgvector_params(
    point: VectorPoint,
    *,
    policy: VectorStorePolicy,
) -> tuple[object, ...]:
    payload = _validate_json_payload(point.payload)
    return (
        point.id,
        point.dense_vector,
        json.dumps(payload, separators=(",", ":")),
        optional_payload_str(payload, policy.namespace_field),
        optional_payload_str(payload, policy.collection_field),
        optional_payload_str(payload, policy.document_id_field),
        optional_payload_str(payload, policy.document_key_field),
        optional_payload_str(payload, policy.content_sha256_field),
        optional_payload_str(payload, policy.processing_version_field),
        optional_payload_str(payload, policy.content_type_field),
        optional_payload_str(payload, policy.source_type_field),
        optional_payload_int(payload, policy.chunk_index_field),
    )


def row_to_search_result(
    row: Mapping[str, object],
    *,
    score: float,
    policy: VectorStorePolicy,
) -> SearchResult:
    return payload_to_result(
        point_id=_required_row_str(row, "id"),
        payload=row_payload(row),
        score=score,
        policy=policy,
    )


def row_payload(row: Mapping[str, object]) -> dict[str, object]:
    value = row.get("payload")
    if isinstance(value, str):
        loaded = json.loads(value)
        if isinstance(loaded, dict):
            return dict(loaded)
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    raise ValueError("pgvector row missing payload object")


def _required_row_str(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"pgvector row missing required string field: {key}")
    return value
