from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Protocol

from rag_core.cli_inputs import cli_safe_error_message
from rag_core.cli_provider_errors import (
    is_provider_bootstrap_error,
    is_provider_error,
    provider_bootstrap_message,
    provider_runtime_message,
)
from rag_core.core_models import RAGCoreConfig
from rag_core.integrations.protocols import SupportsRetrieveContext


class McpCore(SupportsRetrieveContext, Protocol):
    async def ensure_ready(self) -> None: ...

    async def close(self) -> None: ...


async def run_mcp_command(
    args: argparse.Namespace,
    *,
    core_factory: Callable[[RAGCoreConfig], McpCore],
) -> int:
    try:
        from mcp.server.stdio import stdio_server
    except ImportError as exc:
        raise SystemExit(
            "rag-core mcp requires the mcp extra: uv sync --extra mcp"
        ) from exc

    from rag_core.integrations.mcp_server import build_mcp_server

    if not args.corpus_id:
        raise ValueError(
            "--corpus-id is required (repeat the flag to query multiple corpora)"
        )
    if args.limit_cap <= 0:
        raise ValueError("--limit-cap must be positive")

    config = RAGCoreConfig.from_cli(args)
    try:
        core = core_factory(config)
    except Exception as exc:
        if is_provider_bootstrap_error(exc):
            raise ValueError(provider_bootstrap_message(exc, action="mcp")) from exc
        raise

    try:
        try:
            await core.ensure_ready()
        except Exception as exc:
            if is_provider_bootstrap_error(exc):
                raise ValueError(provider_bootstrap_message(exc, action="mcp")) from exc
            raise ValueError(cli_safe_error_message(exc, action="mcp")) from exc

        server = build_mcp_server(
            core,
            namespace=args.namespace,
            corpus_ids=list(args.corpus_id),
            rerank=args.rerank,
            limit_cap=args.limit_cap,
        )
        try:
            async with stdio_server() as (read_stream, write_stream):
                await server.run(
                    read_stream,
                    write_stream,
                    server.create_initialization_options(),
                )
        except Exception as exc:
            if is_provider_error(exc) or is_provider_bootstrap_error(exc):
                raise ValueError(provider_runtime_message(exc, action="mcp")) from exc
            raise
    finally:
        await core.close()
    return 0


__all__ = ["run_mcp_command"]
