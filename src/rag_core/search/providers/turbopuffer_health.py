"""TurboPuffer health payload helpers."""

from __future__ import annotations


def _build_healthy_health(*, namespace: str, metadata: object) -> dict[str, object]:
    index = getattr(metadata, "index", None)
    return {
        "healthy": True,
        "backend": "turbopuffer",
        "namespace": namespace,
        "points_count": getattr(metadata, "approx_row_count", None),
        "logical_bytes": getattr(metadata, "approx_logical_bytes", None),
        "index_status": getattr(index, "status", None),
    }


def _build_unhealthy_health(
    *, namespace: str, exc: Exception
) -> dict[str, object]:
    return {
        "healthy": False,
        "backend": "turbopuffer",
        "namespace": namespace,
        "error": type(exc).__name__,
    }
