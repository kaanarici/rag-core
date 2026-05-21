"""Qdrant health payload helpers."""

from __future__ import annotations

import hashlib
import logging
from typing import Protocol

from rag_core.core_runtime import describe_query_plan_capabilities
from rag_core.search.types import QueryPlanCapabilities

from .qdrant_collection import extract_sparse_vector_names
from .query_plan_capabilities import QDRANT_QUERY_PLAN_CAPABILITIES
from .qdrant_shared import _KNOWN_SPARSE_VECTOR_NAMES, WriteLatencyTracker


class _QdrantHealthClient(Protocol):
    async def get_collection(self, *, collection_name: str) -> object: ...


async def check_qdrant_health(
    *,
    client: _QdrantHealthClient,
    collection_name: str,
    dimensions: int,
    latency: WriteLatencyTracker,
    logger: logging.Logger,
) -> dict[str, object]:
    health = _build_base_health(
        collection_name=collection_name,
        dimensions=dimensions,
    )
    try:
        info = await client.get_collection(collection_name=collection_name)
    except Exception as exc:
        logger.warning(
            "Qdrant health check failed: backend=qdrant error_type=%s "
            "collection_fingerprint=%s",
            type(exc).__name__,
            _collection_fingerprint(collection_name),
        )
        return _build_unhealthy_health(base_health=health, exc=exc)
    return _build_healthy_health(
        base_health=health,
        collection_info=info,
        latency=latency,
    )


def _collection_fingerprint(collection_name: str) -> str:
    return hashlib.sha256(collection_name.encode("utf-8")).hexdigest()[:12]


def _build_base_health(*, collection_name: str, dimensions: int) -> dict[str, object]:
    return {
        "healthy": False,
        "backend": "qdrant",
        "collection": collection_name,
        "dimensions": dimensions,
    }


def _build_healthy_health(
    *,
    base_health: dict[str, object],
    collection_info: object,
    latency: WriteLatencyTracker,
) -> dict[str, object]:
    health = dict(base_health)
    health["healthy"] = True
    health["points_count"] = getattr(collection_info, "points_count", None)

    raw_status = getattr(collection_info, "status", None)
    if raw_status is None:
        health["status"] = "unknown"
    else:
        health["status"] = (
            raw_status.value if hasattr(raw_status, "value") else str(raw_status)
        )

    optimizer_ok = _extract_optimizer_ok(
        getattr(collection_info, "optimizer_status", None)
    )
    if optimizer_ok is not None:
        health["optimizer_ok"] = optimizer_ok
    health["query_plan"] = describe_query_plan_capabilities(
        _collection_query_plan_capabilities(collection_info)
    )

    health["write_latency_p50"] = latency.p50
    health["write_latency_p95"] = latency.p95
    health["write_latency_samples"] = latency.sample_count
    return health


def _collection_query_plan_capabilities(collection_info: object) -> QueryPlanCapabilities:
    sparse_names = extract_sparse_vector_names(collection_info)
    known_sparse_names = (
        sparse_names & _KNOWN_SPARSE_VECTOR_NAMES if sparse_names is not None else None
    )
    if not known_sparse_names:
        return QueryPlanCapabilities(dense=True)
    return QDRANT_QUERY_PLAN_CAPABILITIES


def _build_unhealthy_health(
    *, base_health: dict[str, object], exc: Exception
) -> dict[str, object]:
    health = dict(base_health)
    health["error"] = type(exc).__name__
    return health


def _extract_optimizer_ok(optimizer_status: object | None) -> bool | None:
    if optimizer_status is None:
        return None

    if hasattr(optimizer_status, "ok"):
        return bool(getattr(optimizer_status, "ok"))

    if hasattr(optimizer_status, "status"):
        raw_status = getattr(optimizer_status, "status")
        status_text = (
            raw_status.value if hasattr(raw_status, "value") else str(raw_status)
        )
        return status_text.lower() in {"ok", "green", "healthy"}

    return None
