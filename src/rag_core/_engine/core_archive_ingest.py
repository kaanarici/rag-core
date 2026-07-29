from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rag_core.ingest.sources.archive import (
    ArchiveLimits,
    ArchiveSourceItem,
    ZipArchiveSourceReader,
)
from rag_core.config.ingest_config import DEFAULT_INGEST_MAX_CONCURRENCY
from rag_core._engine.core_archive_runner import (
    ArchiveIngestCore,
    ArchiveIngestRecord,
    ingest_archive_items,
)
from rag_core.documents.exception_names import exception_type
from rag_core.ingest.lifecycle import IngestBatchLifecycle
from rag_core.manifest.persistence import (
    ManifestReconciliation,
    ManifestSource,
    read_entries,
    reconcile_entries,
    validate_manifest_scope,
)
from rag_core.ingest.local.models import (
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
    collection: str,
    metadata: dict[str, str] | None = None,
    force_reindex: bool = False,
    max_concurrency: int = DEFAULT_INGEST_MAX_CONCURRENCY,
    limits: ArchiveLimits | None = None,
    manifest_dir: str | Path | None = None,
    event_sink: "EventSink | None" = None,
) -> LocalIngestResult:
    validate_manifest_scope(namespace, collection)
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
            collection=collection,
            manifest_dir=Path(manifest_dir),
        )
        if manifest_dir is not None
        else None
    )
    lifecycle = IngestBatchLifecycle[ArchiveIngestRecord](
        event_sink=event_sink,
        namespace=namespace,
        collection=collection,
        planned_count=plan.item_count,
        is_success=lambda record: isinstance(record, LocalIngestSuccess),
        error_type=exception_type,
    )
    lifecycle.started()
    batch_failed = False
    try:
        records = await ingest_archive_items(
            core=core,
            items=plan.items,
            lifecycle=lifecycle,
            namespace=namespace,
            collection=collection,
            metadata=metadata,
            force_reindex=force_reindex,
            max_concurrency=max_concurrency,
            limits=resolved_limits,
            reconciliation=reconciliation,
        )
        return LocalIngestResult(
            namespace=namespace,
            collection=collection,
            records=tuple(records),
        )
    except Exception as exc:
        batch_failed = True
        lifecycle.failed(error=exc)
        raise
    finally:
        if not batch_failed:
            lifecycle.completed()


def reconcile_archive_source_plan(
    items: tuple[ArchiveSourceItem, ...],
    *,
    namespace: str,
    collection: str,
    manifest_dir: Path,
) -> ManifestReconciliation:
    entries = read_entries(
        manifest_dir,
        namespace=namespace,
        collection=collection,
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
