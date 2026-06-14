from __future__ import annotations

from rag_core.core_models import IngestedDocument
from rag_core.local_ingest_models import (
    LocalIngestFailure,
    LocalIngestSuccess,
    LocalManifestStatus,
)
from rag_core.safe_messages import safe_error_message
from rag_core.local_sources import LocalSourceItem


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
        error=safe_error_message(exc, action="ingest"),
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


__all__ = [
    "event_error_type",
    "failed_local_ingest_record",
    "failed_local_source_record",
    "successful_local_ingest_record",
]
