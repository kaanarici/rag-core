"""ZIP ingestion with bounded bytes in flight and one read per member."""

from __future__ import annotations

from rag_core._async import run_sync
import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Protocol
from rag_core.config import INGEST_SOURCE_TYPE_ARCHIVE
from rag_core.config.ingest_config import DEFAULT_INGEST_MAX_CONCURRENCY
from rag_core.file_io import compute_content_sha256
from rag_core.core_models import IngestedDocument
from rag_core.documents.exception_names import exception_type
from rag_core.ingest.batch import BatchEvents, bounded_map
from rag_core.ingest.local import (
    LocalIngestFailure,
    LocalIngestSuccess,
    LocalIngestResult,
)
from rag_core.ingest.sources.archive import (
    ArchiveLimits,
    ArchiveSourceItem,
    ZipArchiveReadSession,
    read_zip_archive,
)
from rag_core.manifest.persistence import validate_manifest_scope
from rag_core.safe_messages import safe_error_message

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink


class ArchiveIngestCore(Protocol):
    async def add_bytes(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        tenant_id: str,
        collection: str,
        document_key: str | None = None,
        path: str | None = None,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        source_type: str | None = None,
    ) -> IngestedDocument: ...


class _InFlightArchiveBytes:
    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._used = 0
        self._condition = asyncio.Condition()

    async def acquire(self, byte_count: int) -> None:
        weight = max(1, byte_count)
        async with self._condition:
            await self._condition.wait_for(lambda: self._used + weight <= self._limit)
            self._used += weight

    async def release(self, byte_count: int) -> None:
        weight = max(1, byte_count)
        async with self._condition:
            self._used -= weight
            self._condition.notify_all()


async def ingest_zip_archive_with_core(
    *,
    core: ArchiveIngestCore,
    archive_path: str | Path,
    tenant_id: str,
    collection: str,
    metadata: dict[str, str] | None = None,
    force_reindex: bool = False,
    max_concurrency: int = DEFAULT_INGEST_MAX_CONCURRENCY,
    limits: ArchiveLimits | None = None,
    event_sink: EventSink | None = None,
) -> LocalIngestResult:
    validate_manifest_scope(tenant_id, collection)
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")
    limits = limits or ArchiveLimits()
    plan = await run_sync(
        read_zip_archive, archive_path, limits=limits, hash_content=False
    )
    if not plan.items:
        raise ValueError(f"no supported files found in archive {str(archive_path)!r}")
    batch = BatchEvents(
        event_sink=event_sink,
        tenant_id=tenant_id,
        collection=collection,
        planned_count=plan.item_count,
    )
    budget = _InFlightArchiveBytes(min(limits.max_entry_bytes, limits.max_total_bytes))
    batch.started()
    try:
        with ZipArchiveReadSession(plan.archive_path, limits=limits) as archive:

            async def ingest(position: int, item: ArchiveSourceItem) -> None:
                digest = item.content_sha256
                error = ""
                try:
                    await budget.acquire(item.byte_count)
                    try:
                        data = await run_sync(archive.read_item, item)
                        digest = compute_content_sha256(data)
                        doc = await core.add_bytes(
                            file_bytes=data,
                            filename=item.filename,
                            mime_type=item.mime_type,
                            tenant_id=tenant_id,
                            collection=collection,
                            document_key=item.document_key,
                            path=item.display_path,
                            metadata=metadata,
                            force_reindex=force_reindex,
                            source_type=INGEST_SOURCE_TYPE_ARCHIVE,
                        )
                    finally:
                        await budget.release(item.byte_count)
                    digest = doc.content_sha256 or digest
                    record: LocalIngestSuccess | LocalIngestFailure = (
                        LocalIngestSuccess(
                            path=item.display_path,
                            document_key=doc.document_key or item.document_key,
                            content_sha256=digest,
                            document_id=doc.document_id,
                            filename=doc.filename,
                            chunk_count=doc.chunk_count,
                            ingest_state=doc.ingest_state,
                            replaced_existing=doc.replaced_existing,
                        )
                    )
                except Exception as exc:
                    error = exception_type(exc)
                    record = LocalIngestFailure(
                        path=item.display_path,
                        document_key=item.document_key,
                        content_sha256=digest,
                        error=safe_error_message(exc, action="ingest"),
                    )
                batch.record(position, record, error=error)

            await bounded_map(plan.items, ingest, max_concurrency=max_concurrency)
        batch.completed()
    except BaseException as exc:
        batch.failed(exc)
        raise
    return LocalIngestResult(
        tenant_id=tenant_id, collection=collection, records=batch.records
    )
