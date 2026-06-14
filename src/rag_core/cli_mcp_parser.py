from __future__ import annotations

import argparse

from rag_core.cli_config_parser import add_config_flags


def add_mcp_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    mcp = subparsers.add_parser(
        "mcp",
        help="Run a scope-bound stdio MCP server (requires the mcp extra).",
    )
    add_config_flags(mcp)
    mcp.add_argument("--namespace", required=True)
    mcp.add_argument(
        "--corpus-id",
        action="append",
        default=[],
        help="Repeatable. At least one corpus must be specified.",
    )
    mcp.add_argument(
        "--limit-cap",
        type=int,
        default=10,
        help="Maximum per-tool result limit exposed to the MCP client. Default: 10.",
    )
    mcp.add_argument(
        "--rerank",
        action="store_true",
        help="Apply the configured reranker to MCP retrieval calls.",
    )
    mcp.description = "Expose read-only rag-core retrieval tools over stdio MCP."
    mcp.formatter_class = argparse.RawDescriptionHelpFormatter
    mcp.epilog = """\
Examples:
  uv sync --extra mcp
  rag-core mcp --namespace acme --corpus-id help --qdrant-location :memory: --embedding-provider demo --embedding-dimensions 64
"""


__all__ = ["add_mcp_command"]
