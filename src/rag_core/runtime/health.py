"""Liveness and readiness payloads for the optional ``serve`` runtime."""

from __future__ import annotations


def liveness_payload() -> dict[str, object]:
    """Process is up; does not touch vector store or embeddings."""
    return {
        "ok": True,
        "status": "ok",
        "live": True,
    }


def readiness_payload(
    *,
    ready: bool,
    checks: dict[str, object],
) -> dict[str, object]:
    """Dependency checks after ``RAGCore.ensure_ready`` / store health."""
    status = "ok" if ready else "degraded"
    return {
        "ok": ready,
        "status": status,
        "ready": ready,
        "live": True,
        "checks": checks,
    }


def readiness_status_code(*, ready: bool) -> int:
    return 200 if ready else 503
