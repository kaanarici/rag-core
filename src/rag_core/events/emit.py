"""Emission helpers used by built-in pipeline stages.

These helpers normalize the "stage emits an event, sink swallows its own
errors, stage error becomes a sanitized stage.error event but the exception
still propagates" contract documented in the events seam.

Time source: ``time.monotonic_ns`` is the canonical event ordering source
(``emitted_at_ns``). It is steady: never goes backward and is immune to clock
skew/NTP, so the audit consumer can rely on it for "which event happened
first" without reasoning about wall-clock jitter. ``time.time_ns`` is the
secondary, human-correlation field (``wall_clock_ns``); it is the value an
operator can paste into a human timeline but it MUST NOT be used for
ordering. Decision recorded here so downstream ranks (12, 13) inherit the
same source rather than reinventing it.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Iterator, cast

from .types import Event, StageError

if TYPE_CHECKING:
    from .sink import EventSink


def _stamp_emission_timestamps(event: Event) -> Event:
    """Fill ``emitted_at_ns``/``wall_clock_ns`` if the emitter left them zero.

    The event dataclass owns the *contract* (the fields exist with default
    ``0``). The emit helper owns the *clock* so emitters cannot accidentally
    construct events with a stale or test-controlled timestamp by forgetting
    to pass one. If the emitter set a non-zero value (e.g. An event replay
    test) it is preserved.
    """
    emitted = getattr(event, "emitted_at_ns", None)
    wall = getattr(event, "wall_clock_ns", None)
    if emitted is None and wall is None:
        return event
    updates: dict[str, int] = {}
    if emitted == 0:
        updates["emitted_at_ns"] = time.monotonic_ns()
    if wall == 0:
        updates["wall_clock_ns"] = time.time_ns()
    if not updates:
        return event
    return cast(Event, replace(cast(Any, event), **updates))


def emit_event(sink: "EventSink | None", event: Event) -> None:
    """Emit ``event`` to ``sink``; swallow any sink-side error.

    Stamps ``emitted_at_ns`` and ``wall_clock_ns`` if the emitter left them
    at the default ``0`` so audit consumers always see a real timestamp on
    the canonical audit subset (see ``AUDIT_EVENT_TYPES``).
    """
    if sink is None:
        return
    stamped = _stamp_emission_timestamps(event)
    try:
        sink.emit(stamped)
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
