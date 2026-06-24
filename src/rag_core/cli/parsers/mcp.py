from __future__ import annotations

import argparse

from rag_core.cli.parsers.config import add_config_flags
from rag_core.cli.parsers.sources import add_collection_filters
from rag_core.scope import DEFAULT_NAMESPACE


def add_mcp_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    mcp = subparsers.add_parser(
        "mcp",
        help="Run a scope-bound stdio MCP server (requires the mcp extra).",
    )
    add_config_flags(mcp)
    mcp.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    add_collection_filters(
        mcp,
        help="Repeatable. At least one collection must be specified.",
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
  rag-core mcp --collection help --qdrant-location :memory: --embedding-provider demo --embedding-dimensions 64
"""


__all__ = ["add_mcp_command"]
