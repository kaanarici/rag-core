from __future__ import annotations

import argparse
import asyncio
import io
import sys
from collections.abc import Awaitable, Callable
from contextlib import redirect_stdout
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from rag_core.cli.commands.demo import run_demo_command
from rag_core.cli.deprecations import warn_deprecated_cli
from rag_core.cli.commands.doctor import run_doctor_command
from rag_core.cli.commands.ingest import run_detected_ingest_command
from rag_core.cli.inputs import cli_safe_error_message
from rag_core.cli.commands.eval import run_local_eval_command
from rag_core.cli.commands.local_search import run_local_search_command
from rag_core.cli.commands.manifest import run_manifest_command
from rag_core.cli.commands.mcp import run_mcp_command
from rag_core.cli.parser import _build_parser
from rag_core.cli.commands.search import run_search_command
from rag_core.cli.commands.serve import run_serve_command
from rag_core.core import Engine
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
        _warn_deprecated_command(args)
        if args.command == "doctor":
            return await run_doctor_command(args, core_factory=Engine)
        if args.command == "demo":
            return await run_demo_command(args)
        if args.command == "local-search":
            return await _run_with_event_sink(
                args,
                lambda event_sink: run_local_search_command(args, event_sink=event_sink),
            )
        if args.command in {"eval", "local-eval"}:
            return await _run_with_event_sink(
                args,
                lambda event_sink: run_local_eval_command(args, event_sink=event_sink),
            )
        if args.command == "manifest":
            return await run_manifest_command(args)
        if args.command in {"add", "ingest"}:
            return await _run_with_event_sink(
                args,
                lambda event_sink: run_detected_ingest_command(
                    args,
                    core_factory=Engine,
                    event_sink=event_sink,
                ),
            )
        if args.command in {"search", "context"}:
            return await _run_with_event_sink(
                args,
                lambda event_sink: run_search_command(
                    args,
                    core_factory=Engine,
                    event_sink=event_sink,
                ),
            )
        if args.command == "mcp":
            return await run_mcp_command(args, core_factory=Engine)
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


def _warn_deprecated_command(args: argparse.Namespace) -> None:
    old = getattr(args, "deprecated_command", None)
    new = getattr(args, "canonical_command", None)
    if isinstance(old, str) and isinstance(new, str):
        warn_deprecated_cli(f"rag-core {old}", f"rag-core {new}")


def _event_sink_from_args(args: argparse.Namespace) -> EventSink | None:
    path = getattr(args, "events_jsonl", None)
    if not path:
        return None
    from rag_core.events.sinks import JsonlSink

    event_path = Path(path)
    try:
        return JsonlSink(event_path)
    except OSError as exc:
        raise CliRuntimeError(
            "events JSONL path is invalid or not writable: "
            f"{event_path}. Choose a file path whose parent is a writable directory."
        ) from exc


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
