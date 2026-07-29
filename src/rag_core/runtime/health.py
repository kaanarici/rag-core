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
    event_sink_status: dict[str, object] | None = None,
) -> dict[str, object]:
    """Dependency checks after ``Engine.ensure_ready`` / store health.

    ``event_sink_status`` mirrors the ``event_sink`` block on
    ``describe_runtime`` and surfaces ``failure_count`` so an operator can see
    swallowed sink errors at the same probe they hit for readiness. A non-zero
    ``failure_count`` sets a top-level ``degraded`` flag but does **not**
    flip ``ready`` to ``False``. Sink-side failures are observable, not
    blocking. Callers that want a hard gate can subscribe to the
    ``event_sink`` block themselves; readiness intentionally stays narrow.
    """
    status = "ok" if ready else "degraded"
    payload: dict[str, object] = {
        "ok": ready,
        "status": status,
        "ready": ready,
        "live": True,
        "checks": checks,
    }
    if event_sink_status is not None:
        payload["event_sink"] = event_sink_status
        failure_count = event_sink_status.get("failure_count", 0)
        if isinstance(failure_count, int) and failure_count > 0:
            payload["degraded"] = True
    return payload


def readiness_status_code(*, ready: bool) -> int:
    return 200 if ready else 503
