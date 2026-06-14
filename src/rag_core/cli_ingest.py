from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from rag_core.cli_archive import run_ingest_archive_command
from rag_core.cli_ingest_output import emit_local_ingest_result
from rag_core.cli_inputs import parse_metadata_fields
from rag_core.cli_provider_errors import (
    is_provider_bootstrap_error,
    provider_bootstrap_message,
)
from rag_core.cli_url_ingest import run_ingest_url_command, run_ingest_urls_command
from rag_core.config.ingest_config import DEFAULT_INGEST_MAX_CONCURRENCY
from rag_core.core_models import RAGCoreConfig
from rag_core.local_ingest import (
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

IngestRoute = Literal["local", "archive", "url", "url-list"]
_ARCHIVE_OPTION_ATTRS = (
    "archive_max_entries",
    "archive_max_entry_bytes",
    "archive_max_total_bytes",
)
_FETCH_OPTION_ATTRS = (
    "fetch_max_bytes",
    "fetch_timeout_seconds",
    "fetch_max_redirects",
    "fetch_allow_http",
    "fetch_allow_private_addresses",
)
_URL_ONLY_OPTION_ATTRS = ("document_id",)
_SINGLE_URL_UNSUPPORTED_OPTION_ATTRS = ("max_concurrency", "plan_json")


async def run_detected_ingest_command(
    args: argparse.Namespace,
    *,
    core_factory: Callable[..., RAGCore],
    event_sink: EventSink | None,
) -> int:
    route = _route_ingest_source(args)
    if route == "archive":
        return await run_ingest_archive_command(
            args,
            core_factory=core_factory,
            event_sink=event_sink,
        )
    if route == "url":
        return await run_ingest_url_command(
            args,
            core_factory=core_factory,
            event_sink=event_sink,
        )
    if route == "url-list":
        return await run_ingest_urls_command(
            args,
            core_factory=core_factory,
            event_sink=event_sink,
        )
    return await run_ingest_command(
        args,
        core_factory=core_factory,
        event_sink=event_sink,
    )


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


def _route_ingest_source(args: argparse.Namespace) -> IngestRoute:
    source = args.path
    if args.url_file is not None:
        if source is not None:
            raise ValueError("--url-list cannot be combined with a source argument")
        _reject_provided_options(args, _ARCHIVE_OPTION_ATTRS, route="URL-list ingest")
        _reject_provided_options(args, _URL_ONLY_OPTION_ATTRS, route="URL-list ingest")
        _set_batch_defaults(args)
        return "url-list"
    if source is None:
        raise ValueError(
            "ingest requires a source path/URL or --url-list <file.txt>"
        )
    raw_source = str(source)
    if _is_url_source(raw_source):
        _reject_provided_options(args, _ARCHIVE_OPTION_ATTRS, route="single-URL ingest")
        _reject_provided_options(
            args,
            _SINGLE_URL_UNSUPPORTED_OPTION_ATTRS,
            route="single-URL ingest",
        )
        args.url = raw_source
        return "url"
    if Path(raw_source).suffix.lower() == ".zip":
        _reject_provided_options(args, _FETCH_OPTION_ATTRS, route="archive ingest")
        _reject_provided_options(args, _URL_ONLY_OPTION_ATTRS, route="archive ingest")
        _set_batch_defaults(args)
        args.archive_path = raw_source
        return "archive"
    if _is_local_source(raw_source):
        _reject_provided_options(args, _FETCH_OPTION_ATTRS, route="local file ingest")
        _reject_provided_options(args, _ARCHIVE_OPTION_ATTRS, route="local file ingest")
        _reject_provided_options(args, _URL_ONLY_OPTION_ATTRS, route="local file ingest")
        _set_batch_defaults(args)
        return "local"
    raise FileNotFoundError(
        "ingest source must be an existing file or directory, a glob, a .zip "
        f"archive path, or an HTTP(S) URL: {raw_source}"
    )


def _is_url_source(source: str) -> bool:
    return "://" in source


def _is_local_source(source: str) -> bool:
    if any(char in source for char in "*?["):
        return True
    return Path(source).exists()


def _set_batch_defaults(args: argparse.Namespace) -> None:
    if args.max_concurrency is None:
        args.max_concurrency = DEFAULT_INGEST_MAX_CONCURRENCY


def _reject_provided_options(
    args: argparse.Namespace,
    attrs: tuple[str, ...],
    *,
    route: str,
) -> None:
    provided = [
        _flag_name(attr)
        for attr in attrs
        if _option_was_provided(args, attr)
    ]
    if provided:
        joined = ", ".join(provided)
        raise ValueError(f"{joined} cannot be used with {route}")


def _flag_name(attr: str) -> str:
    return f"--{attr.replace('_', '-')}"


def _option_was_provided(args: argparse.Namespace, attr: str) -> bool:
    value = getattr(args, attr, None)
    if attr == "plan_json":
        return value is True
    return value is not None
