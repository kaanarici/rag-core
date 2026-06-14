from __future__ import annotations

import asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Protocol

from rag_core.cli_provider_errors import is_provider_bootstrap_error
from rag_core.core_models import IngestedDocument
from rag_core._engine.core_file_io import read_file_bytes
from rag_core.ingest_batch_lifecycle import (
    IngestBatchLifecycle,
    IngestBatchProgressPayload,
)
from rag_core.ingest_progress_statuses import (
    INGEST_PROGRESS_FAILED,
    INGEST_PROGRESS_SUCCEEDED,
    IngestProgressStatus,
)
from rag_core.local_ingest_models import (
    LocalIngestFailure,
    LocalIngestPlan,
    LocalIngestRequest,
    LocalIngestSuccess,
)
from rag_core.local_ingest_manifest import (
    manifest_status_for_content,
    manifest_status_for_document,
    source_reconciliation_by_key,
)
from rag_core.local_ingest_records import (
    event_error_type as event_error_type,
    failed_local_ingest_record,
    failed_local_source_record,
    successful_local_ingest_record,
)
from rag_core.manifest_persistence import ManifestReconciliation
from rag_core.local_sources import LocalSourceItem


class LocalIngestCore(Protocol):
    async def ensure_ready(self) -> None: ...

    async def ingest_file(
        self,
        file_path: Path,
        *,
        namespace: str,
        corpus_id: str,
        document_key: str,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        pre_read_bytes: bytes | None = None,
    ) -> IngestedDocument: ...

    async def close(self) -> None: ...


LocalIngestCoreFactory = Callable[[], LocalIngestCore]


@dataclass(frozen=True)
class LocalIngestAborted(Exception):
    cause: Exception
    records: tuple[LocalIngestSuccess | LocalIngestFailure, ...]


async def ingest_local_documents(
    core: LocalIngestCore,
    plan: LocalIngestPlan,
    request: LocalIngestRequest,
    *,
    lifecycle: IngestBatchLifecycle[LocalIngestSuccess | LocalIngestFailure],
    reconciliation: ManifestReconciliation | None = None,
) -> list[LocalIngestSuccess | LocalIngestFailure]:
    if not plan.documents:
        return []
    concurrency = min(request.max_concurrency, plan.document_count)
    document_iter = iter(enumerate(plan.documents))
    document_iter_lock = asyncio.Lock()
    reconciliation_by_key = source_reconciliation_by_key(reconciliation)

    async def ingest_document(position: int, document: LocalSourceItem) -> None:
        manifest_status, manifest_reason = manifest_status_for_document(
            document,
            reconciliation_by_key,
        )
        if document.source_error:
            record: LocalIngestSuccess | LocalIngestFailure
            record = failed_local_source_record(
                document=document,
                manifest_status=manifest_status,
                manifest_reason=manifest_reason,
            )
            actual_content_sha256 = document.content_sha256
            status: IngestProgressStatus = INGEST_PROGRESS_FAILED
            ingest_state = ""
            progress_error = "SourceReadError"
        else:
            try:
                # Read once; pass bytes to avoid a second disk read in ingest_file.
                # If the file changed since discovery, we still use the fresh bytes.
                # The identity hash inside core derives from the actual bytes either way.
                file_bytes = await read_file_bytes(document.path)
                ingested = await core.ingest_file(
                    document.path,
                    namespace=plan.namespace,
                    corpus_id=plan.corpus_id,
                    document_key=document.document_key,
                    metadata=request.metadata,
                    force_reindex=request.force_reindex,
                    pre_read_bytes=file_bytes,
                )
            except Exception as exc:  # noqa: BLE001 - keep the batch running.
                if is_provider_bootstrap_error(exc):
                    partial = await lifecycle.records_snapshot()
                    raise LocalIngestAborted(exc, partial) from exc
                record = failed_local_ingest_record(
                    document=document,
                    exc=exc,
                    manifest_status=manifest_status,
                    manifest_reason=manifest_reason,
                )
                actual_content_sha256 = document.content_sha256
                status = INGEST_PROGRESS_FAILED
                ingest_state = ""
                progress_error = event_error_type(exc)
            else:
                actual_content_sha256 = ingested.content_sha256 or document.content_sha256
                manifest_status, manifest_reason = manifest_status_for_content(
                    document_key=document.document_key,
                    content_sha256=actual_content_sha256,
                    reconciliation_by_key=reconciliation_by_key,
                )
                record = successful_local_ingest_record(
                    document=document,
                    ingested=ingested,
                    content_sha256=actual_content_sha256,
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
                filename=document.path.name,
                document_key=document.document_key,
                content_sha256=actual_content_sha256 or document.content_sha256 or "",
                manifest_status=manifest_status,
                manifest_reason=manifest_reason,
                status=status,
                ingest_state=ingest_state,
                error=progress_error,
            ),
        )

    async def worker() -> None:
        while True:
            async with document_iter_lock:
                try:
                    position, document = next(document_iter)
                except StopIteration:
                    return
            await ingest_document(position, document)

    tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return list(lifecycle.records)


__all__ = [
    "LocalIngestCore",
    "LocalIngestCoreFactory",
    "event_error_type",
    "ingest_local_documents",
]
