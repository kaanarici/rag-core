"""Built-in EventSink implementations.

NoOpSink is the default. Wiring an event sink is opt-in, and callers that do
not pass one see byte-identical behavior to the no-events build.

``JsonlSink`` is a developer-convenience sink for local inspection, debugging,
and tests; it is **not** the canonical audit copy. ``rag-core`` does not
provide a compliance-grade audit transport. That responsibility belongs to
the embedding consumer. Use ``OpenTelemetrySink`` or a user-supplied sink
implementation when audit durability matters.
"""

from __future__ import annotations

import errno
import json
import logging
from pathlib import Path
from threading import Lock

from typing import Any, TextIO

from rag_core.private_files import append_private_text, open_private_append_handle

from .sink import EventSink
from .sink_payloads import event_to_jsonl_dict, event_to_otel_attributes, summarize_event
from .types import Event

_DEFAULT_LOGGER = logging.getLogger("rag_core.events")
NOOP_EVENT_SINK_PROVIDER = "none"
LOGGING_EVENT_SINK_PROVIDER = "logging"
JSONL_EVENT_SINK_PROVIDER = "jsonl"
BUFFER_EVENT_SINK_PROVIDER = "buffer"
MULTI_EVENT_SINK_PROVIDER = "multi"
OPENTELEMETRY_EVENT_SINK_PROVIDER = "opentelemetry"


class _FailureCounter:
    def __init__(self) -> None:
        self._failure_count = 0
        self._failure_lock = Lock()

    @property
    def failure_count(self) -> int:
        with self._failure_lock:
            return self._failure_count

    def _record_failure(self) -> None:
        with self._failure_lock:
            self._failure_count += 1

    def _record_failures(self, count: int) -> None:
        if count <= 0:
            return
        with self._failure_lock:
            self._failure_count += count


class NoOpSink:
    """Drop every event. Default sink when no observer is wired."""

    provider_name = NOOP_EVENT_SINK_PROVIDER

    def emit(self, event: Event) -> None:
        return None


class LoggingSink(_FailureCounter):
    """Forward each event to stdlib ``logging`` at the configured level."""

    provider_name = LOGGING_EVENT_SINK_PROVIDER

    def __init__(
        self,
        logger: logging.Logger | None = None,
        level: int = logging.INFO,
    ) -> None:
        super().__init__()
        self._logger = logger or _DEFAULT_LOGGER
        self._level = level

    def emit(self, event: Event) -> None:
        try:
            self._logger.log(
                self._level,
                "%s %s",
                event.event_type,
                summarize_event(event),
            )
        except Exception:
            self._record_failure()
            # Sinks must not raise.
            pass


class JsonlSink(_FailureCounter):
    """Append each event as one JSON line to ``path``.

    Developer-convenience sink: useful for local debugging, replaying a run
    through a JSONL pipeline, and tests. It is **not** an audit sink. The
    file is not fsynced per write, the path is not durably replicated, and
    nothing here enforces tamper-evidence. If the embedding consumer needs an
    audit copy with retention/integrity guarantees, ship events to its own
    append-only target (e.g. Postgres + S3 object lock) via a user-supplied
    ``EventSink`` adapter or ``OpenTelemetrySink``; rag-core deliberately
    does not ship that target.
    """

    provider_name = JSONL_EVENT_SINK_PROVIDER

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self._path = Path(path)
        self._lock = Lock()
        self._handle: TextIO | None = None
        self._prepare_path()

    def emit(self, event: Event) -> None:
        try:
            payload = json.dumps(
                event_to_jsonl_dict(event),
                allow_nan=False,
                default=str,
            )
            with self._lock:
                if self._handle is None:
                    self._handle = open_private_append_handle(self._path, reject_symlink=True)
                try:
                    self._handle.write(payload + "\n")
                    self._handle.flush()
                except Exception:
                    try:
                        self._handle.close()
                    except Exception:
                        pass
                    self._handle = None
                    raise
        except Exception:
            self._record_failure()
            pass

    def close(self) -> None:
        """Flush and close the underlying file handle.

        Idempotent: safe to call multiple times or when no handle is open.
        A subsequent ``emit`` will lazily reopen the handle.
        """
        with self._lock:
            if self._handle is not None:
                try:
                    self._handle.flush()
                    self._handle.close()
                except Exception:
                    pass
                self._handle = None

    def _prepare_path(self) -> None:
        if self._path.exists() and self._path.is_dir():
            raise ValueError(f"events JSONL path must be a file, not a directory: {self._path}")
        if self._path.is_symlink():
            raise ValueError(f"events JSONL path must not be a symlink: {self._path}")
        try:
            append_private_text(self._path, "", reject_symlink=True)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise ValueError(
                    f"events JSONL path must not be a symlink: {self._path}"
                ) from exc
            raise


class MultiSink(_FailureCounter):
    """Fan an event out to a fixed set of underlying sinks."""

    provider_name = MULTI_EVENT_SINK_PROVIDER

    def __init__(self, *sinks: EventSink) -> None:
        super().__init__()
        self._sinks: tuple[EventSink, ...] = sinks

    def emit(self, event: Event) -> None:
        for sink in self._sinks:
            before_failure_count = _sink_failure_count(sink)
            try:
                sink.emit(event)
            except Exception:
                self._record_failure()
                # One bad sink must not poison the others.
                continue
            after_failure_count = _sink_failure_count(sink)
            if (
                before_failure_count is not None
                and after_failure_count is not None
            ):
                self._record_failures(after_failure_count - before_failure_count)


class EventBuffer(_FailureCounter):
    """Keep every event in memory. Useful for tests and quick inspection."""

    provider_name = BUFFER_EVENT_SINK_PROVIDER

    def __init__(self) -> None:
        super().__init__()
        self.events: list[Event] = []
        self._lock = Lock()

    def emit(self, event: Event) -> None:
        try:
            with self._lock:
                self.events.append(event)
        except Exception:
            self._record_failure()
            pass

    def clear(self) -> None:
        with self._lock:
            self.events.clear()

    def by_type(self, event_type: str) -> list[Event]:
        with self._lock:
            return [event for event in self.events if event.event_type == event_type]


class OpenTelemetrySink(_FailureCounter):
    """Emit events as OpenTelemetry span events on the current span.

    Event names use the ``event_type`` literal directly. Payload fields become
    ``rag_core.*`` attributes with OpenTelemetry-safe primitive values.
    High-risk identifiers and raw messages are omitted by default.
    Requires ``opentelemetry-api``; raises ``ImportError`` at construction time
    if it's missing rather than at module import.
    """

    provider_name = OPENTELEMETRY_EVENT_SINK_PROVIDER

    def __init__(self, *, include_sensitive_attributes: bool = False) -> None:
        super().__init__()
        try:
            import importlib

            otel_trace = importlib.import_module("opentelemetry.trace")
        except ImportError as exc:
            raise ImportError(
                "opentelemetry is required for OpenTelemetrySink. "
                "Install it with: pip install 'rag-core[opentelemetry]' "
                "or pip install opentelemetry-api"
            ) from exc
        self._otel_trace: Any = otel_trace
        self._tracer: Any = otel_trace.get_tracer("rag_core.events")
        self._include_sensitive_attributes = include_sensitive_attributes

    def emit(self, event: Event) -> None:
        try:
            span = self._otel_trace.get_current_span()
            attributes = event_to_otel_attributes(
                event,
                include_sensitive_attributes=self._include_sensitive_attributes,
            )
            is_recording = getattr(span, "is_recording", None)
            if callable(is_recording) and not is_recording():
                with self._tracer.start_as_current_span(
                    event.event_type,
                    attributes=attributes,
                ):
                    return None
            span.add_event(event.event_type, attributes=attributes)
        except Exception:
            self._record_failure()
            pass


def _sink_failure_count(sink: EventSink) -> int | None:
    failure_count = getattr(sink, "failure_count", None)
    if not isinstance(failure_count, bool) and isinstance(failure_count, int):
        return failure_count
    return None


def describe_event_sink_status(sink: EventSink | None) -> dict[str, object]:
    """Project an event sink to its runtime-observable status.

    ``kind`` is the sink's ``provider_name`` (or the class name when the sink
    is user-supplied and does not declare one); ``failure_count`` is the
    total number of swallowed sink errors observed so far. For ``MultiSink``
    the per-child failure deltas already roll up into the parent counter, so
    a single read on the top-level sink is the canonical number.

    Returns the same shape regardless of sink type: ``"kind": "none"`` +
    ``"failure_count": 0`` for a missing or noop sink, so downstream
    diagnostics (``describe_runtime``, readiness payload) can assume the
    field exists without branching on configuration.
    """
    if sink is None:
        return {"kind": NOOP_EVENT_SINK_PROVIDER, "failure_count": 0}
    kind = getattr(sink, "provider_name", None) or type(sink).__name__
    failure_count = _sink_failure_count(sink) or 0
    return {"kind": str(kind), "failure_count": failure_count}


DEFAULT_EVENT_SINK_PROVIDER = NOOP_EVENT_SINK_PROVIDER
EVENT_SINK_PROVIDER_ORDER = (
    DEFAULT_EVENT_SINK_PROVIDER,
    LOGGING_EVENT_SINK_PROVIDER,
    JSONL_EVENT_SINK_PROVIDER,
    BUFFER_EVENT_SINK_PROVIDER,
    MULTI_EVENT_SINK_PROVIDER,
    OPENTELEMETRY_EVENT_SINK_PROVIDER,
)
