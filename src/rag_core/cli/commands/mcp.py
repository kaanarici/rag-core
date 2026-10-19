from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import TYPE_CHECKING

from rag_core.core import Engine
from rag_core.cli.config import config_from_args

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink


async def run_mcp_command(
    args: argparse.Namespace,
    *,
    core_factory: Callable[..., Engine],
    event_sink: EventSink | None = None,
) -> int:
    try:
        from mcp.server.stdio import stdio_server
    except ImportError as exc:
        raise SystemExit(
            "rag-core mcp requires the mcp extra: uv sync --extra mcp"
        ) from exc

    from rag_core.integrations.mcp_server import build_mcp_server

    if not args.collection:
        raise ValueError(
            "--collection is required (repeat the flag for multiple collections)"
        )
    if args.limit_cap <= 0:
        raise ValueError("--limit-cap must be positive")

    config = config_from_args(args)
    async with core_factory(config, event_sink=event_sink) as core:
        server = build_mcp_server(
            core,
            tenant_id=args.tenant_id,
            collections=list(args.collection),
            rerank=False,
            limit_cap=args.limit_cap,
        )
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream, write_stream, server.create_initialization_options()
            )
    return 0


__all__ = ["run_mcp_command"]
