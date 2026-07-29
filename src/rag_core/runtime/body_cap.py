"""Per-request body-size cap for the optional HTTP runtime.

Pure ASGI middleware so it can wrap ``receive`` and tally chunked-transfer
bytes as they arrive. The ``Content-Length`` fast path is incomplete because
omitted / chunked requests bypass it. Read-only verbs (GET/HEAD/OPTIONS) are
skipped because Starlette never reads a body for them.

The module lives outside ``runtime/app.py`` so the app stays under the
architecture-pressure size threshold and the middleware contract has a
single owner.
"""

from __future__ import annotations

from typing import Any

from rag_core.runtime.errors import api_error

# Mirrored from ``runtime.app`` so the two modules stay decoupled. The exact
# string lives in the response header surface and the audit log lines.
_HEADER_REQUEST_ID = "x-request-id"
_SCOPE_REQUEST_ID = "rag_core_request_id"


def _scope_request_id(scope: Any) -> str | None:
    """Read the minted request_id off the ASGI scope state dict.

    The outer ``_RequestIdMiddleware`` stamps the id onto ``request.state``,
    which Starlette persists at ``scope['state']``, so this pure-ASGI inner
    middleware can read it from there without rebuilding a ``Request``.
    """
    state = scope.get("state") if isinstance(scope, dict) else None
    if isinstance(state, dict):
        value = state.get(_SCOPE_REQUEST_ID)
        if isinstance(value, str):
            return value
    return None


async def _send_cap_error(
    send: Any,
    *,
    code: str,
    message: str,
    status_code: int,
    request_id: str | None,
    details: dict[str, object] | None = None,
) -> None:
    """Emit the cap error response via the raw ASGI send channel."""
    response = api_error(
        code=code,
        message=message,
        status_code=status_code,
        details=details,
    )
    if request_id is not None:
        response.headers[_HEADER_REQUEST_ID] = request_id
    await response(
        {"type": "http", "method": "POST"},
        _noop_receive,
        send,
    )


async def _noop_receive() -> dict[str, object]:
    return {"type": "http.disconnect"}


class BodyCapMiddleware:
    """Reject requests over ``cap_bytes`` before parsing the body.

    Two paths:

    1. Content-Length declared: rejected up front (fast path).
    2. Content-Length absent (chunked transfer or omitted): the wrapped
       receive accumulates the running body size as each ``http.request``
       frame arrives and short-circuits with a 413 once the cap is exceeded.

    Backpressure is preserved. The wrapper does not buffer the body, it
    only counts bytes per frame and forwards the original message unchanged
    until the cap trips.
    """

    def __init__(self, app: Any, *, cap_bytes: int) -> None:
        self._app = app
        self._cap = cap_bytes

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return
        method = scope.get("method", "").upper()
        if method in {"GET", "HEAD", "OPTIONS"}:
            await self._app(scope, receive, send)
            return

        request_id = _scope_request_id(scope)
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", ())
        }
        declared = headers.get("content-length")
        if declared is not None:
            try:
                declared_bytes = int(declared)
            except ValueError:
                await _send_cap_error(
                    send,
                    code="invalid_request",
                    message="Content-Length must be an integer",
                    status_code=400,
                    request_id=request_id,
                )
                return
            if declared_bytes > self._cap:
                await _send_cap_error(
                    send,
                    code="payload_too_large",
                    message="request body exceeds configured cap",
                    status_code=413,
                    request_id=request_id,
                    details={"max_bytes": self._cap},
                )
                return
            await self._app(scope, receive, send)
            return

        # Chunked or omitted Content-Length: tally bytes as frames arrive.
        running = 0
        cap = self._cap
        tripped = False

        async def capped_receive() -> Any:
            nonlocal running, tripped
            message = await receive()
            if tripped:
                return message
            if message.get("type") == "http.request":
                running += len(message.get("body", b"") or b"")
                if running > cap:
                    tripped = True
                    # Drain ``more_body`` frames to honor backpressure but
                    # signal end-of-stream to the downstream app; the outer
                    # ``capped_send`` then short-circuits to a 413.
                    return {
                        "type": "http.request",
                        "body": b"",
                        "more_body": False,
                    }
            return message

        cap_tripped_response_sent = False

        async def capped_send(message: Any) -> None:
            nonlocal cap_tripped_response_sent
            if tripped and not cap_tripped_response_sent:
                cap_tripped_response_sent = True
                await _send_cap_error(
                    send,
                    code="payload_too_large",
                    message="request body exceeds configured cap",
                    status_code=413,
                    request_id=request_id,
                    details={"max_bytes": cap},
                )
                return
            if tripped:
                return
            await send(message)

        await self._app(scope, capped_receive, capped_send)


__all__ = ["BodyCapMiddleware"]
