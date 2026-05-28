"""HTTP runtime defaults importable by the base CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Final

DEFAULT_RUNTIME_HOST: Final[str] = "127.0.0.1"
DEFAULT_RUNTIME_JOB_DB_PATH: Final[Path] = Path(".rag-core/runtime/jobs.sqlite3")
DEFAULT_RUNTIME_JOB_DB_PATH_ENV: Final[str] = "RAG_CORE_RUNTIME_JOB_DB_PATH"
DEFAULT_RUNTIME_PORT: Final[int] = 8787

__all__ = [
    "DEFAULT_RUNTIME_HOST",
    "DEFAULT_RUNTIME_JOB_DB_PATH",
    "DEFAULT_RUNTIME_JOB_DB_PATH_ENV",
    "DEFAULT_RUNTIME_PORT",
]
