from __future__ import annotations

import argparse

from rag_core.cli.parsers.config import add_config_flags
from rag_core.runtime_defaults import (
    DEFAULT_RUNTIME_HOST,
    DEFAULT_RUNTIME_INGEST_CONCURRENCY,
    DEFAULT_RUNTIME_LIMIT_CONCURRENCY,
    DEFAULT_RUNTIME_MAX_BODY_BYTES,
    DEFAULT_RUNTIME_PORT,
)


def add_serve_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    serve = subparsers.add_parser(
        "serve",
        help="Run the optional HTTP runtime (requires the runtime extra).",
    )
    add_config_flags(serve)
    serve.add_argument("--host", default=DEFAULT_RUNTIME_HOST)
    serve.add_argument("--port", type=int, default=DEFAULT_RUNTIME_PORT)
    serve.add_argument(
        "--ingest-root",
        action="append",
        default=[],
        help=(
            "Allow POST /v1/ingest paths under this server-local root. "
            "Repeatable; defaults to the current working directory when omitted."
        ),
    )
    serve.add_argument(
        "--unix-socket",
        type=str,
        default=None,
        help=(
            "Bind to a UNIX domain socket instead of host:port. "
            "Mutually exclusive with --host/--port."
        ),
    )
    serve.add_argument(
        "--bind-non-loopback",
        action="store_true",
        default=False,
        help=(
            "Allow binding to a non-loopback host. Without this flag the "
            "server refuses to bind anything except 127.0.0.1, ::1, or "
            "--unix-socket. Required to expose the runtime on a container "
            "network or LAN."
        ),
    )
    serve.add_argument(
        "--max-body-bytes",
        type=int,
        default=DEFAULT_RUNTIME_MAX_BODY_BYTES,
        help=(
            "Maximum accepted request body size in bytes. Requests over the "
            f"cap are refused with HTTP 413. Default: {DEFAULT_RUNTIME_MAX_BODY_BYTES}."
        ),
    )
    serve.add_argument(
        "--ingest-concurrency",
        type=int,
        default=DEFAULT_RUNTIME_INGEST_CONCURRENCY,
        help=(
            "Max concurrent in-flight ingest requests. Additional requests get "
            f"HTTP 503 code='busy'. Default: {DEFAULT_RUNTIME_INGEST_CONCURRENCY}."
        ),
    )
    serve.add_argument(
        "--limit-concurrency",
        type=int,
        default=DEFAULT_RUNTIME_LIMIT_CONCURRENCY,
        help=(
            "Uvicorn-level concurrency ceiling for all routes. "
            f"Default: {DEFAULT_RUNTIME_LIMIT_CONCURRENCY}."
        ),
    )
    serve.description = (
        "Expose health, runtime, ingest, search, and context retrieval over HTTP."
    )
    serve.formatter_class = argparse.RawDescriptionHelpFormatter
    serve.epilog = f"""\
Examples:
  uv sync --extra runtime
  rag-core serve --qdrant-location :memory: --embedding-provider demo --embedding-dimensions 64
  rag-core serve --ingest-root /srv/docs --qdrant-url http://127.0.0.1:6333 --embedding-provider openai --embedding-model text-embedding-3-small
  curl -s http://{DEFAULT_RUNTIME_HOST}:{DEFAULT_RUNTIME_PORT}/health
  See https://kaanarici.github.io/rag-core/docs/self-host for compose + ingest/search curls.
"""


__all__ = ["add_serve_command"]
