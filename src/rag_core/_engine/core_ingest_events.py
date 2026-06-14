from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from rag_core.events.emit import emit_event
from rag_core.events.types import (
    AuditContext,
    Event,
    IndexDeleted,
    IngestCompleted,
    IngestSkipped,
    IngestStarted,
)

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink


class _IngestCorrelationSink:
    """Stamp ``ingest_id`` and caller-supplied audit context onto every event.

    Mirrors ``_SearchCorrelationSink``. The caller supplies an ``ingest_id``
    (the gateway's per-document or per-batch identifier, not minted here so
    a batch can correlate fanned-out per-document events back to the
    originating batch). ``actor`` and ``request_id`` come from
    ``AuditContext``.
    """

    def __init__(
        self,
        sink: "EventSink",
        ingest_id: str,
        audit_context: AuditContext | None,
    ) -> None:
        self._sink = sink
        self._ingest_id = ingest_id
        self._audit_context = audit_context

    def emit(self, event: "Event") -> None:
        updates: dict[str, Any] = {}
        if (
            self._ingest_id
            and hasattr(event, "ingest_id")
            and not getattr(event, "ingest_id")
        ):
            updates["ingest_id"] = self._ingest_id
        ctx = self._audit_context
        if ctx is not None:
            if ctx.actor and hasattr(event, "actor") and not getattr(event, "actor"):
                updates["actor"] = ctx.actor
            if (
                ctx.request_id
                and hasattr(event, "request_id")
                and not getattr(event, "request_id")
            ):
                updates["request_id"] = ctx.request_id
        if updates:
            event = cast("Event", replace(cast(Any, event), **updates))
        self._sink.emit(event)


def emit_ingest_started(
    sink: "EventSink | None",
    *,
    namespace: str,
    corpus_id: str,
    document_id: str,
    filename: str,
    mime_type: str,
    content_sha256: str,
) -> None:
    emit_event(
        sink,
        IngestStarted(
            namespace=namespace,
            corpus_id=corpus_id,
            document_id=document_id,
            filename=filename,
            mime_type=mime_type,
            content_sha256=content_sha256,
        ),
    )


def emit_ingest_skipped(
    sink: "EventSink | None",
    *,
    namespace: str,
    corpus_id: str,
    document_id: str,
) -> None:
    emit_event(
        sink,
        IngestSkipped(
            namespace=namespace,
            corpus_id=corpus_id,
            document_id=document_id,
            reason="content_unchanged",
        ),
    )


def emit_ingest_completed(
    sink: "EventSink | None",
    *,
    namespace: str,
    corpus_id: str,
    document_id: str,
    chunk_count: int,
    duration_ms: float,
) -> None:
    emit_event(
        sink,
        IngestCompleted(
            namespace=namespace,
            corpus_id=corpus_id,
            document_id=document_id,
            chunk_count=chunk_count,
            duration_ms=duration_ms,
        ),
    )


def emit_index_deleted(
    sink: "EventSink | None",
    *,
    namespace: str,
    corpus_id: str,
    document_id: str,
) -> None:
    emit_event(
        sink,
        IndexDeleted(
            namespace=namespace,
            corpus_id=corpus_id,
            document_id=document_id,
        ),
    )


def maybe_wrap_with_ingest_correlation(
    sink: "EventSink | None",
    *,
    ingest_id: str | None,
    audit_context: "AuditContext | None",
) -> "EventSink | None":
    """Return a correlation-wrapped sink only when audit context was supplied.

    Returning the unwrapped sink when no correlation was provided keeps the
    hot path (no-context callers) free of a per-emit ``replace`` allocation.
    """
    if sink is None or not (ingest_id or audit_context):
        return sink
    return _IngestCorrelationSink(sink, ingest_id or "", audit_context)


__all__ = [
    "_IngestCorrelationSink",
    "emit_index_deleted",
    "emit_ingest_completed",
    "emit_ingest_skipped",
    "emit_ingest_started",
    "maybe_wrap_with_ingest_correlation",
]
