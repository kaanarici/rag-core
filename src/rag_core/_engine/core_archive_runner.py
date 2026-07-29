from __future__ import annotations

import asyncio
from typing import Protocol

from rag_core.ingest.sources.archive import ArchiveLimits, ArchiveSourceItem
from rag_core.config import INGEST_SOURCE_TYPE_ARCHIVE
from rag_core.core_models import IngestedDocument
from rag_core.documents.exception_names import exception_type
from rag_core.ingest.lifecycle import (
    IngestBatchLifecycle,
    IngestBatchProgressPayload,
)
from rag_core.ingest.progress import (
    INGEST_PROGRESS_FAILED,
    INGEST_PROGRESS_SUCCEEDED,
    IngestProgressStatus,
)
from rag_core.ingest.local.models import (
    LocalIngestFailure,
    LocalIngestSuccess,
)
from rag_core.ingest.local.manifest import (
    manifest_status_for_content,
    source_reconciliation_by_key,
)
from rag_core.manifest.persistence import ManifestReconciliation
from rag_core.safe_messages import safe_error_message


ArchiveIngestRecord = LocalIngestSuccess | LocalIngestFailure


class ArchiveIngestCore(Protocol):
    async def add_bytes(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        namespace: str,
        collection: str,
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
    lifecycle: IngestBatchLifecycle[ArchiveIngestRecord],
    namespace: str,
    collection: str,
    metadata: dict[str, str] | None,
    force_reindex: bool,
    max_concurrency: int,
    limits: ArchiveLimits,
    reconciliation: ManifestReconciliation | None,
) -> list[ArchiveIngestRecord]:
    if not items:
        return []
    concurrency = min(max_concurrency, len(items))
    item_iter = iter(enumerate(items))
    item_iter_lock = asyncio.Lock()
    reconciliation_by_key = source_reconciliation_by_key(reconciliation)

    async def ingest_item(position: int, item: ArchiveSourceItem) -> None:
        manifest_status, manifest_reason = manifest_status_for_content(
            document_key=item.document_key,
            content_sha256=item.content_sha256,
            reconciliation_by_key=reconciliation_by_key,
        )
        try:
            ingested = await core.add_bytes(
                file_bytes=item.member_bytes,
                filename=item.filename,
                mime_type=item.mime_type,
                namespace=namespace,
                collection=collection,
                document_key=item.document_key,
                path=item.display_path,
                metadata=metadata,
                force_reindex=force_reindex,
                source_type=INGEST_SOURCE_TYPE_ARCHIVE,
            )
        except Exception as exc:  # noqa: BLE001 - record failure and continue.
            record: ArchiveIngestRecord = LocalIngestFailure(
                path=item.display_path,
                document_key=item.document_key,
                content_sha256=item.content_sha256,
                error=safe_error_message(exc, action="ingest"),
                manifest_status=manifest_status,
                manifest_reason=manifest_reason,
            )
            status: IngestProgressStatus = INGEST_PROGRESS_FAILED
            ingest_state = ""
            content_sha256 = item.content_sha256
            progress_error = exception_type(exc)
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
            status = INGEST_PROGRESS_SUCCEEDED
            ingest_state = ingested.ingest_state
            progress_error = ""

        await lifecycle.record(
            position=position,
            record=record,
            progress=IngestBatchProgressPayload(
                filename=item.display_path,
                document_key=record.document_key,
                content_sha256=content_sha256 or "",
                manifest_status=manifest_status,
                manifest_reason=manifest_reason,
                status=status,
                ingest_state=ingest_state,
                error=progress_error,
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
    return list(lifecycle.records)
