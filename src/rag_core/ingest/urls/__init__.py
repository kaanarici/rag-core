from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .api import (
        REMOTE_INGEST_MAX_CONCURRENCY_CAP as REMOTE_INGEST_MAX_CONCURRENCY_CAP,
        RemoteUrlIngestFailure as RemoteUrlIngestFailure,
        RemoteUrlIngestPlan as RemoteUrlIngestPlan,
        RemoteUrlIngestRequest as RemoteUrlIngestRequest,
        RemoteUrlIngestResult as RemoteUrlIngestResult,
        RemoteUrlIngestSuccess as RemoteUrlIngestSuccess,
        build_remote_url_ingest_plan as build_remote_url_ingest_plan,
        reconcile_remote_url_ingest_plan as reconcile_remote_url_ingest_plan,
        run_remote_url_ingest as run_remote_url_ingest,
        run_remote_url_ingest_with_core as run_remote_url_ingest_with_core,
    )

__all__ = [
    "REMOTE_INGEST_MAX_CONCURRENCY_CAP",
    "RemoteUrlIngestFailure",
    "RemoteUrlIngestPlan",
    "RemoteUrlIngestRequest",
    "RemoteUrlIngestResult",
    "RemoteUrlIngestSuccess",
    "build_remote_url_ingest_plan",
    "reconcile_remote_url_ingest_plan",
    "run_remote_url_ingest",
    "run_remote_url_ingest_with_core",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        value = getattr(import_module("rag_core.ingest.urls.api"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'rag_core.ingest.urls' has no attribute {name!r}")
