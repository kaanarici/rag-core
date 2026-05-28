from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rag_core.cli_core_runtime import run_with_ready_core
from rag_core.cli_remote_fetch import (
    discovery_max_sitemap_fetches,
    discovery_max_urls,
    fetch_limits_from_args,
    fetch_policy_from_args,
)
from rag_core.cli_inputs import parse_metadata_fields
from rag_core.cli_provider_errors import (
    ProviderCliError,
    is_provider_error,
    is_provider_bootstrap_error,
    provider_bootstrap_message,
)
from rag_core.cli_remote_output import (
    emit_ingested_document,
    emit_remote_discovery,
    emit_remote_url_ingest,
)
from rag_core.config import INGEST_SOURCE_TYPE_URL
from rag_core.core_config_cli import with_ingest_source_type
from rag_core.core_models import RAGCoreConfig
from rag_core.fetching import FetchError
from rag_core.fetch_security import validate_fetch_url
from rag_core.manifest_persistence import validate_manifest_scope
from rag_core.remote_discovery import (
    RemoteDiscovery,
    RemoteDiscoveryReader,
    redacted_url_file_lines,
    write_raw_discovered_url_file,
    write_redacted_discovered_url_file,
)
from rag_core.remote_discovery_models import (
    DEFAULT_REMOTE_LLMS_TXT_MAX_URLS,
    DEFAULT_REMOTE_SITEMAP_MAX_URLS,
    REMOTE_DISCOVERY_CLI_KIND_LLMS_TXT,
    REMOTE_DISCOVERY_CLI_KIND_SITEMAP,
)
from rag_core.remote_ingest import (
    build_remote_url_ingest_plan,
    reconcile_remote_url_ingest_plan,
    run_remote_url_ingest,
)
from rag_core.remote_ingest_models import (
    RemoteUrlIngestRequest,
)

if TYPE_CHECKING:
    from rag_core.core import RAGCore
    from rag_core.events.sink import EventSink
    from rag_core.core_models import IngestedDocument


async def run_ingest_url_command(
    args: argparse.Namespace,
    *,
    core_factory: Callable[..., RAGCore],
    event_sink: EventSink | None,
) -> int:
    validate_manifest_scope(args.namespace, args.corpus_id)
    fetch_policy = fetch_policy_from_args(args)
    fetch_limits = fetch_limits_from_args(args)
    validate_fetch_url(args.url, policy=fetch_policy)
    metadata = parse_metadata_fields(args.metadata)
    config = with_ingest_source_type(
        RAGCoreConfig.from_cli(args, manifest_dir=Path(args.manifest_dir)),
        source_type=INGEST_SOURCE_TYPE_URL,
    )

    async def run_ingest_url(core: RAGCore) -> "IngestedDocument":
        try:
            return await core.ingest_url(
                args.url,
                namespace=args.namespace,
                corpus_id=args.corpus_id,
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
    core_factory: Callable[..., RAGCore],
    event_sink: EventSink | None,
) -> int:
    validate_manifest_scope(args.namespace, args.corpus_id)
    fetch_policy = fetch_policy_from_args(args)
    fetch_limits = fetch_limits_from_args(args)
    metadata = parse_metadata_fields(args.metadata)
    request = RemoteUrlIngestRequest(
        url_file=args.url_file,
        namespace=args.namespace,
        corpus_id=args.corpus_id,
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
        RAGCoreConfig.from_cli(args, manifest_dir=Path(args.manifest_dir)),
        source_type=INGEST_SOURCE_TYPE_URL,
    )

    def url_core_factory() -> RAGCore:
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


async def run_discover_remote_command(
    args: argparse.Namespace,
    *,
    reader_factory: Callable[..., RemoteDiscoveryReader],
) -> int:
    if args.output_url_file_raw_queries and not args.output_url_file:
        raise ValueError(
            "--output-url-file-raw-queries requires --output-url-file"
        )
    fetch_policy = fetch_policy_from_args(args)
    fetch_limits = fetch_limits_from_args(args)
    validate_fetch_url(args.url, policy=fetch_policy)
    try:
        if args.kind == REMOTE_DISCOVERY_CLI_KIND_SITEMAP:
            max_urls = discovery_max_urls(
                args.max_urls,
                default=DEFAULT_REMOTE_SITEMAP_MAX_URLS,
            )
            max_sitemap_fetches = discovery_max_sitemap_fetches(
                args.max_sitemap_fetches
            )
            reader = reader_factory(policy=fetch_policy, limits=fetch_limits)
            discovery = reader.read_sitemap(
                args.url,
                max_urls=max_urls,
                max_sitemap_fetches=max_sitemap_fetches,
            )
        elif args.kind == REMOTE_DISCOVERY_CLI_KIND_LLMS_TXT:
            max_urls = discovery_max_urls(
                args.max_urls,
                default=DEFAULT_REMOTE_LLMS_TXT_MAX_URLS,
            )
            reader = reader_factory(policy=fetch_policy, limits=fetch_limits)
            discovery = reader.read_llms_txt(args.url, max_urls=max_urls)
        else:
            raise ValueError(f"unsupported remote discovery kind: {args.kind}")
    except (FetchError, ValueError):
        raise
    except Exception as exc:
        raise ValueError("remote discovery failed") from exc
    payload = discovery.to_payload()
    if args.output_url_file:
        url_file = _write_discovered_url_file(
            discovery,
            Path(args.output_url_file),
            raw=bool(args.output_url_file_raw_queries),
        )
        payload["url_file"] = str(url_file.path)
    emit_remote_discovery(payload, as_json=args.json)
    return 0


@dataclass(frozen=True)
class _DiscoveredUrlFileResult:
    path: Path


def _write_discovered_url_file(
    discovery: RemoteDiscovery,
    path: Path,
    *,
    raw: bool,
) -> _DiscoveredUrlFileResult:
    if raw:
        return _DiscoveredUrlFileResult(write_raw_discovered_url_file(discovery, path))
    lines = _cli_redacted_url_file_lines(discovery)
    return _DiscoveredUrlFileResult(write_redacted_discovered_url_file(list(lines), path))


def _cli_redacted_url_file_lines(discovery: RemoteDiscovery) -> tuple[str, ...]:
    try:
        return redacted_url_file_lines(discovery)
    except ValueError as exc:
        message = str(exc)
        if "query-bearing" in message:
            message = message.replace(
                "use write_raw_discovered_url_file()",
                "pass --output-url-file-raw-queries",
            )
            raise ValueError(message) from exc
        raise ValueError(
            "redacted URL output is not safe for URL-file ingestion"
        ) from exc
