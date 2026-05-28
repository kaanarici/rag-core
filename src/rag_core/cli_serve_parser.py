from __future__ import annotations

import argparse
from pathlib import Path

from rag_core.cli_config_parser import add_config_flags, env_or_default
from rag_core.runtime_defaults import (
    DEFAULT_RUNTIME_HOST,
    DEFAULT_RUNTIME_JOB_DB_PATH,
    DEFAULT_RUNTIME_JOB_DB_PATH_ENV,
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
        "--job-db-path",
        type=Path,
        default=Path(
            env_or_default(
                DEFAULT_RUNTIME_JOB_DB_PATH_ENV,
                DEFAULT_RUNTIME_JOB_DB_PATH.as_posix(),
            )
        ),
        help=(
            "SQLite path for ingest job status persistence. "
            f"Default: {DEFAULT_RUNTIME_JOB_DB_PATH} "
            f"(or ${DEFAULT_RUNTIME_JOB_DB_PATH_ENV})."
        ),
    )
    serve.add_argument(
        "--ingest-root",
        action="append",
        default=[],
        help=(
            "Allow POST /v1/ingest paths under this server-local root. "
            "Repeatable; defaults to the current working directory when omitted."
        ),
    )
    serve.description = (
        "Expose health, runtime, ingest jobs, search, and retrieve-context over HTTP."
    )
    serve.formatter_class = argparse.RawDescriptionHelpFormatter
    serve.epilog = f"""\
Examples:
  uv sync --extra runtime
  rag-core serve --qdrant-location :memory: --embedding-provider demo --embedding-dimensions 64
  rag-core serve --job-db-path /var/lib/rag-core/jobs.sqlite3 --ingest-root /srv/docs
  rag-core serve --ingest-root /srv/docs --qdrant-url http://127.0.0.1:6333 --embedding-provider openai --embedding-model text-embedding-3-small --embedding-dimensions 1536
  curl -s http://{DEFAULT_RUNTIME_HOST}:{DEFAULT_RUNTIME_PORT}/health
  See docs/self-host.md for compose + ingest/search curls.
"""
