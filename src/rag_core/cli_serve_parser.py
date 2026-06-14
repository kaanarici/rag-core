from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Final

from rag_core.cli_config_parser import add_config_flags, env_or_default
from rag_core.runtime_defaults import (
    DEFAULT_RUNTIME_HOST,
    DEFAULT_RUNTIME_INGEST_CONCURRENCY,
    DEFAULT_RUNTIME_JOB_DB_PATH,
    DEFAULT_RUNTIME_JOB_DB_PATH_ENV,
    DEFAULT_RUNTIME_LIMIT_CONCURRENCY,
    DEFAULT_RUNTIME_MAX_BODY_BYTES,
    DEFAULT_RUNTIME_PORT,
)

JOB_RETENTION_SECONDS_ENV: Final[str] = "RAG_CORE_RUNTIME_JOB_RETENTION_SECONDS"


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
        "--job-retention-seconds",
        type=_positive_float,
        default=env_or_default(JOB_RETENTION_SECONDS_ENV, "") or None,
        help=(
            "Seconds to retain completed/failed ingest job rows. "
            f"Default: unbounded (or ${JOB_RETENTION_SECONDS_ENV})."
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
            "Max concurrent in-flight ingest jobs. Additional requests get "
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
        "Expose health, runtime, ingest jobs, search, and context retrieval over HTTP."
    )
    serve.formatter_class = argparse.RawDescriptionHelpFormatter
    serve.epilog = f"""\
Examples:
  uv sync --extra runtime
  rag-core serve --qdrant-location :memory: --embedding-provider demo --embedding-dimensions 64
  rag-core serve --job-db-path /var/lib/rag-core/jobs.sqlite3 --ingest-root /srv/docs
  rag-core serve --ingest-root /srv/docs --qdrant-url http://127.0.0.1:6333 --embedding-provider openai --embedding-model text-embedding-3-small
  curl -s http://{DEFAULT_RUNTIME_HOST}:{DEFAULT_RUNTIME_PORT}/health
  See https://kaanarici.github.io/rag-core/docs/self-host for compose + ingest/search curls.
"""


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive finite number") from exc
    if parsed <= 0 or not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed
