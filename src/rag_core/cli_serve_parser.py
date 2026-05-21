from __future__ import annotations

import argparse

from rag_core.cli_config_parser import add_config_flags
def add_serve_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    serve = subparsers.add_parser(
        "serve",
        help="Run the optional HTTP runtime (requires the runtime extra).",
    )
    add_config_flags(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8787)
    serve.description = (
        "Expose health, runtime, ingest jobs, search, and retrieve-context over HTTP."
    )
    serve.formatter_class = argparse.RawDescriptionHelpFormatter
    serve.epilog = """\
Examples:
  uv sync --extra runtime
  rag-core serve --qdrant-location :memory: --embedding-provider demo --embedding-dimensions 64
  rag-core serve --qdrant-url http://127.0.0.1:6333 --embedding-provider openai --embedding-model text-embedding-3-small
  curl -s http://127.0.0.1:8787/health
  See docs/self-host/quickstart.md for compose + ingest/search curls.
"""
