from __future__ import annotations

from typing import TYPE_CHECKING

from rag_core.events.emit import emit_event
from rag_core.events.sink import CorrelatedSink
from rag_core.events.types import (
    AuditContext,
    IndexDeleted,
    IngestCompleted,
    IngestSkipped,
    IngestStarted,
)

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink


def emit_ingest_started(
    sink: "EventSink | None",
    *,
    tenant_id: str,
    collection: str,
    document_id: str,
    filename: str,
    mime_type: str,
    content_sha256: str,
) -> None:
    emit_event(
        sink,
        IngestStarted(
            tenant_id=tenant_id,
            collection=collection,
            document_id=document_id,
            filename=filename,
            mime_type=mime_type,
            content_sha256=content_sha256,
        ),
    )


def emit_ingest_skipped(
    sink: "EventSink | None",
    *,
    tenant_id: str,
    collection: str,
    document_id: str,
) -> None:
    emit_event(
        sink,
        IngestSkipped(
            tenant_id=tenant_id,
            collection=collection,
            document_id=document_id,
            reason="content_unchanged",
        ),
    )


def emit_ingest_completed(
    sink: "EventSink | None",
    *,
    tenant_id: str,
    collection: str,
    document_id: str,
    chunk_count: int,
    duration_ms: float,
) -> None:
    emit_event(
        sink,
        IngestCompleted(
            tenant_id=tenant_id,
            collection=collection,
            document_id=document_id,
            chunk_count=chunk_count,
            duration_ms=duration_ms,
        ),
    )


def emit_index_deleted(
    sink: "EventSink | None",
    *,
    tenant_id: str,
    collection: str,
    document_id: str,
) -> None:
    emit_event(
        sink,
        IndexDeleted(
            tenant_id=tenant_id,
            collection=collection,
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
    return CorrelatedSink(sink, ingest_id=ingest_id or "", audit_context=audit_context)


__all__ = [
    "emit_index_deleted",
    "emit_ingest_completed",
    "emit_ingest_skipped",
    "emit_ingest_started",
    "maybe_wrap_with_ingest_correlation",
]
