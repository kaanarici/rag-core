from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .api import (
        LocalContextCore as LocalContextCore,
        LocalContextCoreFactory as LocalContextCoreFactory,
        LocalIngestCore as LocalIngestCore,
        LocalIngestCoreFactory as LocalIngestCoreFactory,
        LocalIngestFailure as LocalIngestFailure,
        LocalIngestPlan as LocalIngestPlan,
        LocalIngestRequest as LocalIngestRequest,
        LocalIngestResult as LocalIngestResult,
        LocalIngestSuccess as LocalIngestSuccess,
        LocalSearchCore as LocalSearchCore,
        LocalSearchCoreFactory as LocalSearchCoreFactory,
        LocalSearchRequest as LocalSearchRequest,
        LocalSearchResult as LocalSearchResult,
        ManifestPreviewRequest as ManifestPreviewRequest,
        ManifestPreviewResult as ManifestPreviewResult,
        build_local_ingest_plan as build_local_ingest_plan,
        default_collection as default_collection,
        discover_local_files as discover_local_files,
        local_search_hit_payload as local_search_hit_payload,
        preview_manifest as preview_manifest,
        reconcile_local_ingest_plan as reconcile_local_ingest_plan,
        run_local_context as run_local_context,
        run_local_ingest as run_local_ingest,
        run_local_ingest_with_core as run_local_ingest_with_core,
        run_local_search as run_local_search,
        validate_supported_local_file as validate_supported_local_file,
    )

__all__ = [
    "LocalIngestCore",
    "LocalIngestCoreFactory",
    "LocalIngestFailure",
    "LocalIngestPlan",
    "LocalIngestRequest",
    "LocalIngestResult",
    "LocalIngestSuccess",
    "LocalContextCore",
    "LocalContextCoreFactory",
    "LocalSearchCore",
    "LocalSearchCoreFactory",
    "LocalSearchRequest",
    "LocalSearchResult",
    "ManifestPreviewRequest",
    "ManifestPreviewResult",
    "build_local_ingest_plan",
    "default_collection",
    "discover_local_files",
    "local_search_hit_payload",
    "preview_manifest",
    "reconcile_local_ingest_plan",
    "run_local_ingest",
    "run_local_ingest_with_core",
    "run_local_context",
    "run_local_search",
    "validate_supported_local_file",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        value = getattr(import_module("rag_core.ingest.local.api"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'rag_core.ingest.local' has no attribute {name!r}")
