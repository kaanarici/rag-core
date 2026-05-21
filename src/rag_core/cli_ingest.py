from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from rag_core.cli_ingest_output import emit_local_ingest_result
from rag_core.cli_inputs import parse_metadata_fields
from rag_core.cli_provider_errors import (
    is_provider_bootstrap_error,
    provider_bootstrap_message,
)
from rag_core.core_models import RAGCoreConfig
from rag_core.local_corpus import (
    build_local_ingest_plan,
    reconcile_local_ingest_plan,
    run_local_ingest,
)
from rag_core.local_ingest_models import (
    LocalIngestRequest,
)

if TYPE_CHECKING:
    from rag_core.core import RAGCore
    from rag_core.events.sink import EventSink


async def run_ingest_command(
    args: argparse.Namespace,
    *,
    core_factory: Callable[..., RAGCore],
    event_sink: EventSink | None,
) -> int:
    metadata = parse_metadata_fields(args.metadata)
    manifest_dir = Path(args.manifest_dir)
    request = LocalIngestRequest(
        path=args.path,
        namespace=args.namespace,
        corpus_id=args.corpus_id,
        metadata=metadata or None,
        force_reindex=args.force_reindex,
        max_concurrency=args.max_concurrency,
    )

    if args.plan_json:
        plan = build_local_ingest_plan(request)
        reconciliation = reconcile_local_ingest_plan(plan, manifest_dir=manifest_dir)
        print(
            json.dumps(
                plan.to_payload(reconciliation=reconciliation),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    def local_core_factory() -> RAGCore:
        config = RAGCoreConfig.from_cli(args, manifest_dir=manifest_dir)
        return core_factory(config, event_sink=event_sink)

    try:
        result = await run_local_ingest(
            request,
            core_factory=local_core_factory,
            event_sink=event_sink,
            manifest_dir=manifest_dir,
        )
    except Exception as exc:
        if is_provider_bootstrap_error(exc):
            raise ValueError(provider_bootstrap_message(exc, action="ingest")) from exc
        raise
    emit_local_ingest_result(result, as_json=args.json)
    return 1 if result.failed_count else 0
