from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Literal, Protocol

from rag_core.archive_sources import ArchiveLimits, ArchiveSourceItem
from rag_core.cli_inputs import cli_safe_error_message
from rag_core.core_models import IngestedDocument
from rag_core.events.emit import emit_event
from rag_core.events.types import IngestBatchProgress
from rag_core.local_ingest_models import (
    LocalIngestFailure,
    LocalIngestSuccess,
)
from rag_core.local_ingest_manifest import (
    manifest_status_for_content,
    source_reconciliation_by_key,
)
from rag_core.manifest_persistence import ManifestReconciliation

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink


class ArchiveIngestCore(Protocol):
    async def ingest_bytes(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        namespace: str,
        corpus_id: str,
        document_key: str | None = None,
        path: str | None = None,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        source_type: str | None = None,
    ) -> IngestedDocument: ...


async def ingest_archive_items(
    *,
    core: ArchiveIngestCore,
    items: tuple[ArchiveSourceItem, ...],
    records: list[LocalIngestSuccess | LocalIngestFailure | None],
    namespace: str,
    corpus_id: str,
    metadata: dict[str, str] | None,
    force_reindex: bool,
    max_concurrency: int,
    limits: ArchiveLimits,
    reconciliation: ManifestReconciliation | None,
    event_sink: "EventSink | None",
) -> None:
    if not items:
        return
    concurrency = min(max_concurrency, len(items))
    item_iter = iter(enumerate(items))
    item_iter_lock = asyncio.Lock()
    progress_lock = asyncio.Lock()
    completed_count = 0
    succeeded_count = 0
    failed_count = 0
    reconciliation_by_key = source_reconciliation_by_key(reconciliation)

    async def ingest_item(position: int, item: ArchiveSourceItem) -> None:
        nonlocal completed_count, succeeded_count, failed_count
        manifest_status, manifest_reason = manifest_status_for_content(
            document_key=item.document_key,
            content_sha256=item.content_sha256,
            reconciliation_by_key=reconciliation_by_key,
        )
        try:
            ingested = await core.ingest_bytes(
                file_bytes=item.member_bytes,
                filename=item.filename,
                mime_type=item.mime_type,
                namespace=namespace,
                corpus_id=corpus_id,
                document_key=item.document_key,
                path=item.display_path,
                metadata=metadata,
                force_reindex=force_reindex,
                source_type="archive",
            )
        except Exception as exc:  # noqa: BLE001 - record failure and continue.
            error_message = cli_safe_error_message(exc, action="ingest")
            record: LocalIngestSuccess | LocalIngestFailure = LocalIngestFailure(
                path=item.display_path,
                document_key=item.document_key,
                content_sha256=item.content_sha256,
                error=error_message,
                manifest_status=manifest_status,
                manifest_reason=manifest_reason,
            )
            status: Literal["succeeded", "failed"] = "failed"
            ingest_state = ""
            content_sha256 = item.content_sha256
            error = type(exc).__name__
        else:
            content_sha256 = ingested.content_sha256 or item.content_sha256
            manifest_status, manifest_reason = manifest_status_for_content(
                document_key=item.document_key,
                content_sha256=content_sha256,
                reconciliation_by_key=reconciliation_by_key,
            )
            record = LocalIngestSuccess(
                path=item.display_path,
                document_key=ingested.document_key or item.document_key,
                content_sha256=content_sha256,
                document_id=ingested.document_id,
                filename=ingested.filename,
                chunk_count=ingested.chunk_count,
                ingest_state=ingested.ingest_state,
                replaced_existing=ingested.replaced_existing,
                manifest_status=manifest_status,
                manifest_reason=manifest_reason,
            )
            status = "succeeded"
            ingest_state = ingested.ingest_state
            error = ""

        async with progress_lock:
            records[position] = record
            completed_count += 1
            if isinstance(record, LocalIngestSuccess):
                succeeded_count += 1
            else:
                failed_count += 1
            emit_event(
                event_sink,
                IngestBatchProgress(
                    namespace=namespace,
                    corpus_id=corpus_id,
                    completed_count=completed_count,
                    planned_count=len(items),
                    succeeded_count=succeeded_count,
                    failed_count=failed_count,
                    current_index=completed_count,
                    filename=item.display_path,
                    document_key=record.document_key,
                    content_sha256=content_sha256,
                    manifest_status=manifest_status,
                    manifest_reason=manifest_reason,
                    status=status,
                    ingest_state=ingest_state,
                    error=error,
                ),
            )

    async def worker() -> None:
        while True:
            async with item_iter_lock:
                try:
                    index, item = next(item_iter)
                except StopIteration:
                    return
            await ingest_item(index, item)

    await asyncio.gather(*(worker() for _ in range(concurrency)))


def compact_archive_records(
    records: list[LocalIngestSuccess | LocalIngestFailure | None],
) -> list[LocalIngestSuccess | LocalIngestFailure]:
    return [record for record in records if record is not None]


def archive_record_counts(
    records: list[LocalIngestSuccess | LocalIngestFailure],
) -> tuple[int, int]:
    succeeded = sum(isinstance(record, LocalIngestSuccess) for record in records)
    return succeeded, len(records) - succeeded
