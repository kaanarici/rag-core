from __future__ import annotations

from typing import TYPE_CHECKING

from rag_core.events.emit import emit_event
from rag_core.events.types import (
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


__all__ = [
    "emit_index_deleted",
    "emit_ingest_completed",
    "emit_ingest_skipped",
    "emit_ingest_started",
]
