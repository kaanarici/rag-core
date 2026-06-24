"""Qdrant payload conversion plus shared constants and write tuning helpers.

Hosts the small qdrant-local constants and the ``WriteLatencyTracker`` /
``compute_write_params`` helpers consumed by the collection, query, write, and
store modules. Those live here (the lowest qdrant module above ``qdrant_filters``)
so the higher-level modules can share them without an import cycle through the
store entry.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from typing import Any, Optional
from uuid import UUID

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as rest

from rag_core.search.document_records import (
    resolve_document_id_from_payload,
    stored_document_record_from_payload,
    validate_document_lookup_inputs,
)
from rag_core.search.policy import VectorStorePolicy
from rag_core.search.request_models import StoredDocumentRecord
from rag_core.search.sparse_channels import (
    KNOWN_SPARSE_CHANNELS,
    PRIMARY_SPARSE_CHANNEL,
    SECONDARY_SPARSE_CHANNEL,
)
from rag_core.search.stored_payload import payload_to_result
from rag_core.search.stored_payload import validate_json_payload as _validate_json_payload
from rag_core.search.vector_models import SearchResult, VectorPoint

from .chunk_lookup import validate_chunk_lookup_inputs
from .qdrant_filters import (
    build_chunk_index_lookup_filter,
    build_document_count_filter,
    build_document_lookup_filter,
)


_DENSE_VECTOR_NAME = ""  # Qdrant default vector name
_PRIMARY_SPARSE_VECTOR_NAME = PRIMARY_SPARSE_CHANNEL
_SECONDARY_SPARSE_VECTOR_NAME = SECONDARY_SPARSE_CHANNEL
_KNOWN_SPARSE_VECTOR_NAMES = KNOWN_SPARSE_CHANNELS
_PREFETCH_LIMIT = 200
_MAX_SPLIT_DEPTH = 6
_SPLIT_PAUSE_SECONDS = 0.2
_SLOW_WRITE_THRESHOLD_SECONDS = 5.0
_LATENCY_WINDOW_SIZE = 100


def compute_write_params(vector_size: int) -> tuple[int, int]:
    """Compute concurrency and batch limits for a given vector dimension."""
    if vector_size >= 3000:
        return 4, 40
    if vector_size >= 1024:
        return 8, 60
    return 16, 100


class WriteLatencyTracker:
    """Tracks P50/P95 write latencies in a fixed-size window."""

    def __init__(self, window_size: int = _LATENCY_WINDOW_SIZE) -> None:
        self._samples: deque[float] = deque(maxlen=window_size)

    def record(self, duration_seconds: float) -> None:
        """Record a write duration sample."""
        self._samples.append(duration_seconds)

    def percentile(self, p: float) -> Optional[float]:
        """Return the p-th percentile or ``None`` when no samples exist."""
        if not self._samples:
            return None
        sorted_samples = sorted(self._samples)
        idx = int(len(sorted_samples) * p / 100.0)
        idx = min(idx, len(sorted_samples) - 1)
        return sorted_samples[idx]

    @property
    def p50(self) -> Optional[float]:
        """Median write latency."""
        return self.percentile(50)

    @property
    def p95(self) -> Optional[float]:
        """95th percentile write latency."""
        return self.percentile(95)

    @property
    def sample_count(self) -> int:
        """Number of recorded samples."""
        return len(self._samples)


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


async def get_qdrant_document_record(
    *,
    client: AsyncQdrantClient,
    collection_name: str,
    namespace: str,
    collection: str,
    document_id: str | None,
    document_key: str | None,
    policy: VectorStorePolicy,
) -> StoredDocumentRecord | None:
    namespace_scoped, collection_scoped = validate_document_lookup_inputs(
        namespace=namespace,
        collection=collection,
        document_id=document_id,
        document_key=document_key,
    )

    records, _ = await client.scroll(
        collection_name=collection_name,
        scroll_filter=build_document_lookup_filter(
            namespace=namespace_scoped,
            collection=collection_scoped,
            document_id=document_id,
            document_key=document_key,
            policy=policy,
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not records:
        return None

    payload = records[0].payload or {}
    resolved_document_id = resolve_document_id_from_payload(
        payload=payload,
        document_id_field=policy.document_id_field,
        fallback_document_id=document_id,
        invalid_message=(
            f"qdrant document record payload field {policy.document_id_field!r} "
            "must be a string"
        ),
    )
    if not resolved_document_id:
        return None
    chunk_count = await client.count(
        collection_name=collection_name,
        count_filter=build_document_count_filter(
            namespace=namespace_scoped,
            collection=collection_scoped,
            document_id=resolved_document_id,
            policy=policy,
        ),
        exact=True,
    )
    return stored_document_record_from_payload(
        payload=payload,
        namespace=namespace_scoped,
        collection=collection_scoped,
        document_id=resolved_document_id,
        chunk_count=_count_result_value(getattr(chunk_count, "count", None)),
        policy=policy,
        invalid_field_message=(
            "qdrant document record payload field {field!r} must be a string"
        ),
    )


async def get_qdrant_chunks_by_index(
    *,
    client: AsyncQdrantClient,
    collection_name: str,
    namespace: str,
    collection: str,
    document_id: str,
    chunk_indices: Sequence[int],
    policy: VectorStorePolicy,
) -> list[SearchResult]:
    namespace_scoped, collection_scoped, document_scoped, indices = (
        validate_chunk_lookup_inputs(
            namespace=namespace,
            collection=collection,
            document_id=document_id,
            chunk_indices=chunk_indices,
        )
    )
    if not indices:
        return []

    records, _ = await client.scroll(
        collection_name=collection_name,
        scroll_filter=build_chunk_index_lookup_filter(
            namespace=namespace_scoped,
            collection=collection_scoped,
            document_id=document_scoped,
            chunk_indices=indices,
            policy=policy,
        ),
        limit=len(indices),
        with_payload=True,
        with_vectors=False,
    )
    results = [_record_to_result(record, policy=policy) for record in records]
    return sorted(results, key=lambda result: result.chunk_index or 0)
