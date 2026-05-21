from __future__ import annotations

import argparse
import asyncio
import io
import sys
from collections.abc import Awaitable, Callable
from contextlib import redirect_stdout
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from rag_core.cli_archive import run_ingest_archive_command
from rag_core.cli_demo import run_demo_command
from rag_core.cli_doctor import run_doctor_command
from rag_core.cli_ingest import run_ingest_command
from rag_core.cli_inputs import cli_safe_error_message
from rag_core.cli_local_search import run_local_search_command
from rag_core.cli_manifest import run_manifest_command, run_manifest_compact_command
from rag_core.cli_parser import _build_parser
from rag_core.cli_search import run_search_command
from rag_core.cli_serve import run_serve_command
from rag_core.cli_remote import (
    run_discover_remote_command,
    run_ingest_url_command,
    run_ingest_urls_command,
)
from rag_core.cli_remote_fetch import remote_discovery_reader as _remote_discovery_reader
from rag_core.core import RAGCore
from rag_core.fetching import FetchError

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink


class CliRuntimeError(RuntimeError):
    pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "serve":
        return run_serve_command(args)
    return asyncio.run(_run_command(args, parser))


async def async_main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "serve":
        return run_serve_command(args)
    return await _run_command(args, parser)


async def _run_command(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    try:
        if args.command == "doctor":
            return await run_doctor_command(args, core_factory=RAGCore)
        if args.command == "demo":
            return await run_demo_command(args)
        if args.command == "local-search":
            return await _run_with_event_sink(
                args,
                lambda event_sink: run_local_search_command(args, event_sink=event_sink),
            )
        if args.command == "manifest":
            return await run_manifest_command(args)
        if args.command == "manifest-compact":
            return await run_manifest_compact_command(args)
        if args.command == "ingest":
            return await _run_with_event_sink(
                args,
                lambda event_sink: run_ingest_command(
                    args,
                    core_factory=RAGCore,
                    event_sink=event_sink,
                ),
            )
        if args.command == "ingest-archive":
            return await _run_with_event_sink(
                args,
                lambda event_sink: run_ingest_archive_command(
                    args,
                    core_factory=RAGCore,
                    event_sink=event_sink,
                ),
            )
        if args.command == "ingest-url":
            return await _run_with_event_sink(
                args,
                lambda event_sink: run_ingest_url_command(
                    args,
                    core_factory=RAGCore,
                    event_sink=event_sink,
                ),
            )
        if args.command == "ingest-urls":
            return await _run_with_event_sink(
                args,
                lambda event_sink: run_ingest_urls_command(
                    args,
                    core_factory=RAGCore,
                    event_sink=event_sink,
                ),
            )
        if args.command == "discover-remote":
            return await run_discover_remote_command(
                args,
                reader_factory=_remote_discovery_reader,
            )
        if args.command in {"search", "retrieve-context"}:
            return await _run_with_event_sink(
                args,
                lambda event_sink: run_search_command(
                    args,
                    core_factory=RAGCore,
                    event_sink=event_sink,
                ),
            )
    except (FetchError, FileNotFoundError, ValueError) as exc:
        parser.exit(
            2,
            f"rag-core: error: {cli_safe_error_message(exc, action=args.command)}\n",
        )
    except CliRuntimeError as exc:
        parser.exit(
            1,
            f"rag-core: error: {cli_safe_error_message(exc, action=args.command)}\n",
        )
    parser.print_help()
    return 1


def _event_sink_from_args(args: argparse.Namespace) -> EventSink | None:
    path = getattr(args, "events_jsonl", None)
    if not path:
        return None
    from rag_core.events.sinks import JsonlSink

    return JsonlSink(Path(path))


async def _run_with_event_sink(
    args: argparse.Namespace,
    run: Callable[["EventSink | None"], Awaitable[int]],
) -> int:
    event_sink = _event_sink_from_args(args)
    if event_sink is None:
        return await run(None)
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = await run(event_sink)
    _raise_event_sink_failures(event_sink)
    sys.stdout.write(stdout.getvalue())
    return exit_code


def _raise_event_sink_failures(event_sink: "EventSink | None") -> None:
    failure_count = getattr(event_sink, "failure_count", 0)
    if isinstance(failure_count, int) and failure_count > 0:
        raise CliRuntimeError(
            f"events JSONL sink failed to write {failure_count} event(s)"
        )
