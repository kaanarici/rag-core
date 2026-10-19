from __future__ import annotations
import argparse
import asyncio
import io
import json
import sys
from collections.abc import Awaitable, Callable, Sequence
from contextlib import redirect_stdout
from pathlib import Path
from typing import TYPE_CHECKING, Any
from rag_core.cli.parser import _build_parser
from rag_core.cli.inputs import parse_metadata_fields
from rag_core.safe_messages import safe_error_message
from rag_core.core import Engine
from rag_core.cli.config import config_from_args
from rag_core.fetching import FetchError

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink
    from rag_core.events.sinks import JsonlSink


class CliRuntimeError(RuntimeError):
    pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_run_command(args, parser))


async def async_main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    return await _run_command(parser.parse_args(argv), parser)


async def _run_command(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> int:
    try:
        return await _run_with_event_sink(args, lambda sink: _dispatch(args, sink))
    except (FetchError, OSError, ValueError, RuntimeError) as exc:
        parser.exit(
            2, f"rag-core: error: {safe_error_message(exc, action=args.command)}\n"
        )
    except Exception as exc:
        from rag_core.provider_errors import is_provider_error, provider_runtime_message

        if is_provider_error(exc):
            parser.exit(
                2,
                f"rag-core: error: {provider_runtime_message(exc, action=args.command)}\n",
            )
        raise


async def _dispatch(args: argparse.Namespace, sink: EventSink | None) -> int:
    if args.command in {"search", "context"}:
        from .commands.search import run_search_command

        return await run_search_command(args, core_factory=Engine, event_sink=sink)
    if args.command == "demo":
        from .commands.demo import run_demo_command

        return await run_demo_command(args)
    if args.command == "eval":
        from .commands.eval import run_local_eval_command

        return await run_local_eval_command(args, core_factory=Engine, event_sink=sink)
    if args.command == "mcp":
        from .commands.mcp import run_mcp_command

        return await run_mcp_command(args, core_factory=Engine, event_sink=sink)
    if args.command == "add":
        return await _add(args, sink)
    if args.command == "doctor":
        config = config_from_args(args)
        async with Engine(config, event_sink=sink) as core:
            payload = {
                "runtime": core.describe_runtime(),
                "health": await core.check_health(),
            }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    raise ValueError(f"Unknown command: {args.command}")


async def _add(args: argparse.Namespace, sink: EventSink | None) -> int:
    from rag_core.ingest.local import LocalIngestRequest, build_local_ingest_plan

    if len(args.collection) != 1:
        raise ValueError("add requires exactly one --collection")
    scope: dict[str, Any] = {
        "tenant_id": args.tenant_id,
        "collection": args.collection[0],
    }
    if args.dry_run:
        plan = build_local_ingest_plan(
            LocalIngestRequest(
                path=args.path,
                tenant_id=args.tenant_id or "default",
                collection=args.collection[0],
                max_files=args.max_files,
            ),
            hash_content=False,
        )
        print(json.dumps(plan.to_payload(), indent=2))
        return 0
    metadata = parse_metadata_fields(args.metadata) or None
    async with Engine(config_from_args(args), event_sink=sink) as core:
        if args.path.startswith(("https://", "http://")):
            from dataclasses import asdict

            result = asdict(
                await core.add_url(
                    args.path,
                    metadata=metadata,
                    force_reindex=args.force_reindex,
                    **scope,
                )
            )
        elif Path(args.path).suffix.lower() == ".zip":
            result = (
                await core.add_archive(
                    args.path,
                    metadata=metadata,
                    force_reindex=args.force_reindex,
                    max_concurrency=args.max_concurrency,
                    **scope,
                )
            ).to_payload()
        else:
            result = (
                await core.add(
                    args.path,
                    metadata=metadata,
                    force_reindex=args.force_reindex,
                    max_files=args.max_files,
                    max_concurrency=args.max_concurrency,
                    **scope,
                )
            ).to_payload()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 1 if result.get("failed_count", 0) else 0


def _event_sink_from_args(args: argparse.Namespace) -> JsonlSink | None:
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
    try:
        # MCP uses stdout for a live protocol; buffering it would deadlock clients.
        if args.command == "mcp":
            exit_code = await run(event_sink)
        else:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = await run(event_sink)
            _raise_event_sink_failures(event_sink)
            sys.stdout.write(stdout.getvalue())
        _raise_event_sink_failures(event_sink)
        return exit_code
    finally:
        event_sink.close()


def _raise_event_sink_failures(event_sink: "EventSink | None") -> None:
    failure_count = getattr(event_sink, "failure_count", 0)
    if isinstance(failure_count, int) and failure_count > 0:
        raise CliRuntimeError(
            f"events JSONL sink failed to write {failure_count} event(s)"
        )
