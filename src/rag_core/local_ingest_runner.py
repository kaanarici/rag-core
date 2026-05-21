from __future__ import annotations

import asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Protocol, Sequence

from rag_core.cli_provider_errors import is_provider_bootstrap_error
from rag_core.core_models import IngestedDocument
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
    LocalIngestProgressStatus,
    emit_local_ingest_progress,
    event_error_type as event_error_type,
    failed_local_ingest_record,
    failed_local_source_record,
    successful_local_ingest_record,
)
from rag_core.manifest_persistence import ManifestReconciliation
from rag_core.sources import LocalSourceItem

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink


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
    event_sink: EventSink | None = None,
    reconciliation: ManifestReconciliation | None = None,
) -> list[LocalIngestSuccess | LocalIngestFailure]:
    records: list[LocalIngestSuccess | LocalIngestFailure | None] = [
        None for _ in plan.documents
    ]
    succeeded_count = 0
    failed_count = 0
    completed_count = 0
    progress_lock = asyncio.Lock()
    if not plan.documents:
        return []
    concurrency = min(request.max_concurrency, plan.document_count)
    document_iter = iter(enumerate(plan.documents))
    document_iter_lock = asyncio.Lock()
    reconciliation_by_key = source_reconciliation_by_key(reconciliation)

    async def ingest_document(position: int, document: LocalSourceItem) -> None:
        nonlocal completed_count, failed_count, succeeded_count
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
            status: LocalIngestProgressStatus = "failed"
            ingest_state = ""
            progress_error = "SourceReadError"
        else:
            try:
                ingested = await core.ingest_file(
                    document.path,
                    namespace=plan.namespace,
                    corpus_id=plan.corpus_id,
                    document_key=document.document_key,
                    metadata=request.metadata,
                    force_reindex=request.force_reindex,
                )
            except Exception as exc:  # noqa: BLE001 - keep the batch running.
                if is_provider_bootstrap_error(exc):
                    async with progress_lock:
                        partial = tuple(record for record in records if record is not None)
                    raise LocalIngestAborted(exc, partial) from exc
                record = failed_local_ingest_record(
                    document=document,
                    exc=exc,
                    manifest_status=manifest_status,
                    manifest_reason=manifest_reason,
                )
                actual_content_sha256 = document.content_sha256
                status = "failed"
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
                status = "succeeded"
                ingest_state = ingested.ingest_state
                progress_error = ""

        async with progress_lock:
            records[position] = record
            completed_count += 1
            if isinstance(record, LocalIngestSuccess):
                succeeded_count += 1
                error = ""
            else:
                failed_count += 1
                error = progress_error
            emit_local_ingest_progress(
                event_sink,
                plan=plan,
                document=document,
                current_index=completed_count,
                completed_count=completed_count,
                succeeded_count=succeeded_count,
                failed_count=failed_count,
                status=status,
                content_sha256=actual_content_sha256,
                manifest_status=manifest_status,
                manifest_reason=manifest_reason,
                ingest_state=ingest_state,
                error=error,
            )

    async def worker() -> None:
        while True:
            async with document_iter_lock:
                try:
                    position, document = next(document_iter)
                except StopIteration:
                    return
            await ingest_document(position, document)

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return [record for record in records if record is not None]


def local_ingest_record_counts(
    records: Sequence[LocalIngestSuccess | LocalIngestFailure],
) -> tuple[int, int]:
    succeeded_count = sum(
        1 for record in records if isinstance(record, LocalIngestSuccess)
    )
    failed_count = sum(
        1 for record in records if isinstance(record, LocalIngestFailure)
    )
    return succeeded_count, failed_count


__all__ = [
    "LocalIngestCore",
    "LocalIngestCoreFactory",
    "event_error_type",
    "ingest_local_documents",
    "local_ingest_record_counts",
]
