"""HTTP runtime defaults importable by the base CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Final

DEFAULT_RUNTIME_HOST: Final[str] = "127.0.0.1"
DEFAULT_RUNTIME_JOB_DB_PATH: Final[Path] = Path(".rag-core/runtime/jobs.sqlite3")
DEFAULT_RUNTIME_JOB_DB_PATH_ENV: Final[str] = "RAG_CORE_RUNTIME_JOB_DB_PATH"
DEFAULT_RUNTIME_PORT: Final[int] = 8787

# Body cap defaults. Ingest bodies carry only the JSON envelope
# ``{"path", "namespace", "corpus_id"}`` (the file content stays on disk and
# is referenced by ``path``), so 4 MiB is more than enough headroom. Control
# plane (search / context retrieval) bodies are tiny: query text plus a few
# scope arrays, so we cap them tighter.
DEFAULT_RUNTIME_MAX_BODY_BYTES: Final[int] = 4 * 1024 * 1024
# Process-wide ingest-job semaphore. Caps concurrent ingest worker tasks so
# a bursty caller cannot exhaust file descriptors / embedder budget. 503 with
# ``code='busy'`` once saturated.
DEFAULT_RUNTIME_INGEST_CONCURRENCY: Final[int] = 8
# Uvicorn ``--limit-concurrency`` ceiling. Higher than the ingest semaphore so
# read paths (search / health / context retrieval) stay responsive while ingest
# is at the per-route cap.
DEFAULT_RUNTIME_LIMIT_CONCURRENCY: Final[int] = 64
# Loopback hosts that don't need ``--bind-non-loopback`` to accept.
LOOPBACK_HOSTS: Final[frozenset[str]] = frozenset(
    {"127.0.0.1", "localhost", "::1"}
)

__all__ = [
    "DEFAULT_RUNTIME_HOST",
    "DEFAULT_RUNTIME_INGEST_CONCURRENCY",
    "DEFAULT_RUNTIME_JOB_DB_PATH",
    "DEFAULT_RUNTIME_JOB_DB_PATH_ENV",
    "DEFAULT_RUNTIME_LIMIT_CONCURRENCY",
    "DEFAULT_RUNTIME_MAX_BODY_BYTES",
    "DEFAULT_RUNTIME_PORT",
    "LOOPBACK_HOSTS",
]
