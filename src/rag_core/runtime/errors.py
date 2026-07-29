"""Stable HTTP error payloads for the optional ``serve`` runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from starlette.responses import JSONResponse


@dataclass(frozen=True)
class RedactedRuntimeError:
    """Sanitized view of an exception for public HTTP/job surfaces.

    ``error_type`` is the exception class name; ``error_code`` is a stable
    machine-readable label callers can branch on (``unhealthy`` for readiness
    failures, ``ingest_failed`` for terminal ingest jobs). Full ``str(exc)``
    text is intentionally absent. It can include SDK message strings or
    licensed source identifiers and must stay in logs/event sinks only.
    """

    error_type: str
    error_code: str


def redact_runtime_error(exc: BaseException, *, error_code: str) -> RedactedRuntimeError:
    """Build a ``RedactedRuntimeError`` from an exception, dropping ``str(exc)``."""
    return RedactedRuntimeError(error_type=type(exc).__name__, error_code=error_code)


class RuntimeRequestError(ValueError):
    """Invalid runtime HTTP request shape or field values."""

    def __init__(
        self,
        *,
        message: str,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


def api_error(
    *,
    code: str,
    message: str,
    status_code: int,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    """Return a JSON error body shared by all runtime routes."""
    error: dict[str, Any] = {"code": code, "message": message}
    if details:
        error["details"] = details
    return JSONResponse({"error": error}, status_code=status_code)


async def parse_json_object(request: Any) -> dict[str, Any] | JSONResponse:
    """Parse a JSON object body or return a 400 ``api_error`` response."""
    try:
        body = await request.json()
    except Exception:
        return api_error(
            code="invalid_json",
            message="Request body must be valid JSON",
            status_code=400,
        )
    if not isinstance(body, dict):
        return api_error(
            code="invalid_request",
            message="Request body must be a JSON object",
            status_code=400,
        )
    return body


__all__ = [
    "RedactedRuntimeError",
    "RuntimeRequestError",
    "api_error",
    "parse_json_object",
    "redact_runtime_error",
]
