from __future__ import annotations

from typing import TYPE_CHECKING

from rag_core.cli_inputs import cli_safe_error_message
from rag_core.core_models import IngestedDocument
from rag_core.events.emit import emit_event
from rag_core.events.types import IngestBatchProgress
from rag_core.ingest_progress_statuses import IngestProgressStatus
from rag_core.local_ingest_models import (
    LocalIngestFailure,
    LocalIngestPlan,
    LocalIngestSuccess,
    LocalManifestStatus,
)
from rag_core.manifest_reconciliation_reasons import MANIFEST_REASON_NOT_CHECKED
from rag_core.manifest_reconciliation_statuses import MANIFEST_STATUS_UNKNOWN
from rag_core.sources import LocalSourceItem

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink


def failed_local_source_record(
    *,
    document: LocalSourceItem,
    manifest_status: LocalManifestStatus,
    manifest_reason: str,
) -> LocalIngestFailure:
    return LocalIngestFailure(
        path=str(document.path),
        document_key=document.document_key,
        content_sha256=document.content_sha256,
        error=document.source_error or "source read failed",
        manifest_status=manifest_status,
        manifest_reason=manifest_reason,
    )


def failed_local_ingest_record(
    *,
    document: LocalSourceItem,
    exc: Exception,
    manifest_status: LocalManifestStatus,
    manifest_reason: str,
) -> LocalIngestFailure:
    return LocalIngestFailure(
        path=str(document.path),
        document_key=document.document_key,
        content_sha256=document.content_sha256,
        error=cli_safe_error_message(exc, action="ingest"),
        manifest_status=manifest_status,
        manifest_reason=manifest_reason,
    )


def successful_local_ingest_record(
    *,
    document: LocalSourceItem,
    ingested: IngestedDocument,
    content_sha256: str | None,
    manifest_status: LocalManifestStatus,
    manifest_reason: str,
) -> LocalIngestSuccess:
    return LocalIngestSuccess(
        path=str(document.path),
        document_key=document.document_key,
        content_sha256=content_sha256,
        document_id=ingested.document_id,
        filename=ingested.filename,
        chunk_count=ingested.chunk_count,
        ingest_state=ingested.ingest_state,
        replaced_existing=ingested.replaced_existing,
        manifest_status=manifest_status,
        manifest_reason=manifest_reason,
    )


def event_error_type(exc: Exception) -> str:
    return type(exc).__name__


def emit_local_ingest_progress(
    event_sink: EventSink | None,
    *,
    plan: LocalIngestPlan,
    document: LocalSourceItem,
    current_index: int,
    completed_count: int,
    succeeded_count: int,
    failed_count: int,
    status: IngestProgressStatus,
    content_sha256: str | None = None,
    manifest_status: LocalManifestStatus = MANIFEST_STATUS_UNKNOWN,
    manifest_reason: str = MANIFEST_REASON_NOT_CHECKED,
    ingest_state: str = "",
    error: str = "",
) -> None:
    emit_event(
        event_sink,
        IngestBatchProgress(
            namespace=plan.namespace,
            corpus_id=plan.corpus_id,
            planned_count=plan.document_count,
            completed_count=completed_count,
            succeeded_count=succeeded_count,
            failed_count=failed_count,
            current_index=current_index,
            filename=document.path.name,
            document_key=document.document_key,
            content_sha256=content_sha256 or document.content_sha256 or "",
            manifest_status=manifest_status,
            manifest_reason=manifest_reason,
            status=status,
            ingest_state=ingest_state,
            error=error,
        ),
    )


__all__ = [
    "emit_local_ingest_progress",
    "event_error_type",
    "failed_local_ingest_record",
    "failed_local_source_record",
    "successful_local_ingest_record",
]
