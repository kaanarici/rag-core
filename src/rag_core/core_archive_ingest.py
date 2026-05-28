from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rag_core.archive_sources import (
    ArchiveLimits,
    ArchiveSourceItem,
    ZipArchiveSourceReader,
)
from rag_core.config.ingest_config import DEFAULT_INGEST_MAX_CONCURRENCY
from rag_core.core_archive_runner import (
    ArchiveIngestCore,
    archive_record_counts,
    compact_archive_records,
    ingest_archive_items,
)
from rag_core.events.emit import emit_event, now_ms
from rag_core.events.types import (
    IngestBatchCompleted,
    IngestBatchFailed,
    IngestBatchStarted,
)
from rag_core.manifest_persistence import (
    ManifestReconciliation,
    ManifestSource,
    read_entries,
    reconcile_entries,
    validate_manifest_scope,
)
from rag_core.local_ingest_models import (
    LocalIngestFailure,
    LocalIngestResult,
    LocalIngestSuccess,
)

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink


async def ingest_zip_archive_with_core(
    *,
    core: ArchiveIngestCore,
    archive_path: str | Path,
    namespace: str,
    corpus_id: str,
    metadata: dict[str, str] | None = None,
    force_reindex: bool = False,
    max_concurrency: int = DEFAULT_INGEST_MAX_CONCURRENCY,
    limits: ArchiveLimits | None = None,
    manifest_dir: str | Path | None = None,
    event_sink: "EventSink | None" = None,
) -> LocalIngestResult:
    validate_manifest_scope(namespace, corpus_id)
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")
    resolved_limits = limits or ArchiveLimits()
    plan = ZipArchiveSourceReader().read(archive_path, limits=resolved_limits)
    if not plan.items:
        raise ValueError(f"no supported files found in archive {str(archive_path)!r}")
    reconciliation = (
        reconcile_archive_source_plan(
            plan.items,
            namespace=namespace,
            corpus_id=corpus_id,
            manifest_dir=Path(manifest_dir),
        )
        if manifest_dir is not None
        else None
    )
    started_ms = now_ms()
    emit_event(
        event_sink,
        IngestBatchStarted(
            namespace=namespace,
            corpus_id=corpus_id,
            planned_count=plan.item_count,
        ),
    )
    records: list[LocalIngestSuccess | LocalIngestFailure | None] = [
        None for _ in plan.items
    ]
    batch_failed = False
    try:
        await ingest_archive_items(
            core=core,
            items=plan.items,
            records=records,
            namespace=namespace,
            corpus_id=corpus_id,
            metadata=metadata,
            force_reindex=force_reindex,
            max_concurrency=max_concurrency,
            limits=resolved_limits,
            reconciliation=reconciliation,
            event_sink=event_sink,
        )
        compact_records = compact_archive_records(records)
        return LocalIngestResult(
            namespace=namespace,
            corpus_id=corpus_id,
            records=tuple(compact_records),
        )
    except Exception as exc:
        batch_failed = True
        completed = compact_archive_records(records)
        succeeded_count, failed_count = archive_record_counts(completed)
        emit_event(
            event_sink,
            IngestBatchFailed(
                namespace=namespace,
                corpus_id=corpus_id,
                planned_count=plan.item_count,
                completed_count=len(completed),
                succeeded_count=succeeded_count,
                failed_count=failed_count,
                duration_ms=now_ms() - started_ms,
                error=type(exc).__name__,
            ),
        )
        raise
    finally:
        if not batch_failed:
            completed = compact_archive_records(records)
            succeeded_count, failed_count = archive_record_counts(completed)
            emit_event(
                event_sink,
                IngestBatchCompleted(
                    namespace=namespace,
                    corpus_id=corpus_id,
                    planned_count=plan.item_count,
                    succeeded_count=succeeded_count,
                    failed_count=failed_count,
                    duration_ms=now_ms() - started_ms,
                ),
            )


def reconcile_archive_source_plan(
    items: tuple[ArchiveSourceItem, ...],
    *,
    namespace: str,
    corpus_id: str,
    manifest_dir: Path,
) -> ManifestReconciliation:
    entries = read_entries(
        manifest_dir,
        namespace=namespace,
        corpus_id=corpus_id,
    )
    return reconcile_entries(
        entries,
        (
            ManifestSource(
                document_key=item.document_key,
                content_sha256=item.content_sha256,
            )
            for item in items
        ),
    )


__all__ = [
    "ArchiveIngestCore",
    "ingest_zip_archive_with_core",
    "reconcile_archive_source_plan",
]
