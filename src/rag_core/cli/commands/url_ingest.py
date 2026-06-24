from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from rag_core._engine.core_config_cli import with_ingest_source_type
from rag_core.cli.core_runtime import run_with_ready_core
from rag_core.cli.fetch_options import fetch_limits_from_args, fetch_policy_from_args
from rag_core.cli.ingest_output import emit_ingested_document, emit_remote_url_ingest
from rag_core.cli.inputs import parse_metadata_fields
from rag_core.provider_errors import (
    ProviderCliError,
    is_provider_bootstrap_error,
    is_provider_error,
    provider_bootstrap_message,
)
from rag_core.config import INGEST_SOURCE_TYPE_URL
from rag_core.core_models import Config
from rag_core.fetch_security import validate_fetch_url
from rag_core.manifest.persistence import validate_manifest_scope
from rag_core.ingest.urls import (
    build_remote_url_ingest_plan,
    reconcile_remote_url_ingest_plan,
    run_remote_url_ingest,
)
from rag_core.ingest.urls.models import RemoteUrlIngestRequest

if TYPE_CHECKING:
    from rag_core.core import Engine
    from rag_core.core_models import IngestedDocument
    from rag_core.events.sink import EventSink


async def run_ingest_url_command(
    args: argparse.Namespace,
    *,
    core_factory: Callable[..., Engine],
    event_sink: EventSink | None,
) -> int:
    validate_manifest_scope(args.namespace, args.collection)
    fetch_policy = fetch_policy_from_args(args)
    fetch_limits = fetch_limits_from_args(args)
    validate_fetch_url(args.url, policy=fetch_policy)
    metadata = parse_metadata_fields(args.metadata)
    config = with_ingest_source_type(
        Config.from_cli(args, manifest_dir=Path(args.manifest_dir)),
        source_type=INGEST_SOURCE_TYPE_URL,
    )

    async def run_ingest_url(core: Engine) -> IngestedDocument:
        try:
            return await core.add_url(
                args.url,
                namespace=args.namespace,
                collection=args.collection,
                document_id=args.document_id,
                metadata=metadata or None,
                force_reindex=args.force_reindex,
                fetch_policy=fetch_policy,
                fetch_limits=fetch_limits,
            )
        except Exception as exc:
            if is_provider_error(exc) or is_provider_bootstrap_error(exc):
                raise
            raise ValueError("remote ingest failed") from exc

    document = await run_with_ready_core(
        core_factory=lambda: core_factory(config, event_sink=event_sink),
        action="ingest",
        run=run_ingest_url,
    )

    emit_ingested_document(document, as_json=args.json)
    return 0


async def run_ingest_urls_command(
    args: argparse.Namespace,
    *,
    core_factory: Callable[..., Engine],
    event_sink: EventSink | None,
) -> int:
    validate_manifest_scope(args.namespace, args.collection)
    fetch_policy = fetch_policy_from_args(args)
    fetch_limits = fetch_limits_from_args(args)
    metadata = parse_metadata_fields(args.metadata)
    request = RemoteUrlIngestRequest(
        url_file=args.url_file,
        namespace=args.namespace,
        collection=args.collection,
        metadata=metadata or None,
        force_reindex=args.force_reindex,
        max_concurrency=args.max_concurrency,
        fetch_policy=fetch_policy,
        fetch_limits=fetch_limits,
    )
    if args.plan_json:
        plan = build_remote_url_ingest_plan(request)
        reconciliation = reconcile_remote_url_ingest_plan(
            plan,
            manifest_dir=Path(args.manifest_dir),
        )
        print(
            json.dumps(
                plan.to_payload(reconciliation=reconciliation),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    config = with_ingest_source_type(
        Config.from_cli(args, manifest_dir=Path(args.manifest_dir)),
        source_type=INGEST_SOURCE_TYPE_URL,
    )

    def url_core_factory() -> Engine:
        return core_factory(
            config,
            event_sink=event_sink,
        )

    try:
        result = await run_remote_url_ingest(
            request,
            core_factory=url_core_factory,
            event_sink=event_sink,
            manifest_dir=Path(args.manifest_dir),
        )
    except Exception as exc:
        if is_provider_bootstrap_error(exc):
            raise ProviderCliError(
                provider_bootstrap_message(exc, action="ingest")
            ) from exc
        raise
    emit_remote_url_ingest(result, as_json=args.json)
    return 1 if result.failed_count else 0
