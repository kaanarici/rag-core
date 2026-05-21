from __future__ import annotations

from collections.abc import Sequence

from rag_core.core_models import IngestedDocument
from rag_core.fetch_security import redact_fetch_url
from rag_core.remote_ingest_models import (
    RemoteManifestStatus,
    RemoteUrlIngestFailure,
    RemoteUrlIngestSuccess,
    RemoteUrlSourceItem,
)


def remote_ingest_success_record(
    *,
    item: RemoteUrlSourceItem,
    ingested: IngestedDocument,
    manifest_status: RemoteManifestStatus,
    manifest_reason: str,
) -> RemoteUrlIngestSuccess:
    return RemoteUrlIngestSuccess(
        requested_url=item.redacted_url,
        source_url=safe_remote_output_source_url(
            ingested.metadata.get("source_url"),
            fallback=item.redacted_url,
        ),
        document_key=ingested.document_key or item.document_key,
        content_sha256=ingested.content_sha256,
        document_id=ingested.document_id,
        filename=ingested.filename,
        chunk_count=ingested.chunk_count,
        ingest_state=ingested.ingest_state,
        replaced_existing=ingested.replaced_existing,
        manifest_status=manifest_status,
        manifest_reason=manifest_reason,
    )


def safe_remote_ingest_error(exc: Exception, item: RemoteUrlSourceItem) -> str:
    return f"{type(exc).__name__} while ingesting {item.redacted_url}"


def remote_ingest_error_type(exc: Exception) -> str:
    return type(exc).__name__


def remote_ingest_record_counts(
    records: Sequence[RemoteUrlIngestSuccess | RemoteUrlIngestFailure],
) -> tuple[int, int]:
    succeeded_count = sum(
        1 for record in records if isinstance(record, RemoteUrlIngestSuccess)
    )
    failed_count = sum(
        1 for record in records if isinstance(record, RemoteUrlIngestFailure)
    )
    return succeeded_count, failed_count


def safe_remote_output_source_url(value: object, *, fallback: str) -> str:
    if not isinstance(value, str):
        return fallback
    redacted = redact_fetch_url(value)
    return fallback if redacted == "<invalid-url>" else redacted
