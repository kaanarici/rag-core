from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from rag_core.cli_core_runtime import run_with_ready_core
from rag_core.archive_sources import (
    ArchiveLimits,
    ArchiveSourceItem,
    ArchiveSourcePlan,
    ZipArchiveSourceReader,
)
from rag_core.cli_ingest_output import emit_local_ingest_result
from rag_core.cli_inputs import parse_metadata_fields
from rag_core.config import INGEST_SOURCE_TYPE_ARCHIVE
from rag_core.core_config_cli import with_ingest_source_type
from rag_core.core_archive_ingest import reconcile_archive_source_plan
from rag_core.core_models import RAGCoreConfig
from rag_core.local_ingest_manifest import (
    manifest_status_for_content,
    source_reconciliation_by_key,
)
from rag_core.manifest_persistence import (
    ManifestReconciliation,
    ManifestReconciliationItem,
    manifest_reconciliation_payload,
)

if TYPE_CHECKING:
    from rag_core.core import RAGCore
    from rag_core.events.sink import EventSink
    from rag_core.local_ingest_models import LocalIngestResult


async def run_ingest_archive_command(
    args: argparse.Namespace,
    *,
    core_factory: Callable[..., RAGCore],
    event_sink: EventSink | None,
) -> int:
    limits = _archive_limits_from_args(args)
    metadata = parse_metadata_fields(args.metadata)
    manifest_dir = Path(args.manifest_dir)

    if args.plan_json:
        plan = ZipArchiveSourceReader().read(args.archive_path, limits=limits)
        reconciliation = reconcile_archive_source_plan(
            plan.items,
            namespace=args.namespace,
            corpus_id=args.corpus_id,
            manifest_dir=manifest_dir,
        )
        print(
            json.dumps(
                _archive_plan_payload(
                    plan,
                    namespace=args.namespace,
                    corpus_id=args.corpus_id,
                    reconciliation=reconciliation,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    config = with_ingest_source_type(
        RAGCoreConfig.from_cli(args, manifest_dir=manifest_dir),
        source_type=INGEST_SOURCE_TYPE_ARCHIVE,
    )

    async def run_archive(core: RAGCore) -> "LocalIngestResult":
        return await core.ingest_archive(
            args.archive_path,
            namespace=args.namespace,
            corpus_id=args.corpus_id,
            metadata=metadata or None,
            force_reindex=args.force_reindex,
            max_concurrency=args.max_concurrency,
            archive_limits=limits,
            manifest_dir=manifest_dir,
        )

    result = await run_with_ready_core(
        core_factory=lambda: core_factory(config, event_sink=event_sink),
        action="ingest",
        run=run_archive,
    )

    emit_local_ingest_result(result, as_json=args.json)
    return 1 if result.failed_count else 0


def _archive_limits_from_args(args: argparse.Namespace) -> ArchiveLimits:
    return ArchiveLimits(
        max_entries=args.archive_max_entries,
        max_entry_bytes=args.archive_max_entry_bytes,
        max_total_bytes=args.archive_max_total_bytes,
    )


def _archive_plan_payload(
    plan: ArchiveSourcePlan,
    *,
    namespace: str,
    corpus_id: str,
    reconciliation: ManifestReconciliation | None,
) -> dict[str, object]:
    reconciliation_by_key = source_reconciliation_by_key(reconciliation)
    payload: dict[str, object] = {
        "source_type": INGEST_SOURCE_TYPE_ARCHIVE,
        "archive_name": plan.archive_path.name,
        "namespace": namespace,
        "corpus_id": corpus_id,
        "planned_count": plan.item_count,
        "items": [
            _archive_item_payload(
                item,
                reconciliation=reconciliation,
                reconciliation_by_key=reconciliation_by_key,
            )
            for item in plan.items
        ],
    }
    if reconciliation is not None:
        payload["reconciliation"] = manifest_reconciliation_payload(reconciliation)
    return payload


def _archive_item_payload(
    item: ArchiveSourceItem,
    *,
    reconciliation: ManifestReconciliation | None,
    reconciliation_by_key: dict[str, ManifestReconciliationItem],
) -> dict[str, object]:
    payload = item.to_payload()
    if reconciliation is None:
        return payload
    manifest_status, manifest_reason = manifest_status_for_content(
        document_key=item.document_key,
        content_sha256=item.content_sha256,
        reconciliation_by_key=reconciliation_by_key,
    )
    return {
        **payload,
        "manifest_status": manifest_status,
        "manifest_reason": manifest_reason,
    }


__all__ = ["run_ingest_archive_command"]
