"""Emission helpers used by built-in pipeline stages.

These helpers normalize the "stage emits an event, sink swallows its own
errors, stage error becomes a sanitized stage.error event but the exception
still propagates" contract documented in the events seam.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

from .types import Event, StageError

if TYPE_CHECKING:
    from .sink import EventSink


def emit_event(sink: "EventSink | None", event: Event) -> None:
    """Emit ``event`` to ``sink``; swallow any sink-side error."""
    if sink is None:
        return
    try:
        sink.emit(event)
    except Exception:
        pass


def now_ms() -> float:
    """Return monotonic milliseconds for duration measurement."""
    return time.perf_counter() * 1000.0


@contextmanager
def stage_guard(
    sink: "EventSink | None",
    *,
    stage: str,
) -> Iterator[None]:
    """Context manager that emits a sanitized StageError, then re-raises."""
    try:
        yield
    except Exception as exc:
        emit_event(
            sink,
            StageError(
                stage=stage,
                error_type=type(exc).__name__,
            ),
        )
        raise
