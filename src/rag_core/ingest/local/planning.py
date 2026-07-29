from __future__ import annotations

from pathlib import Path

from rag_core.documents.converters.format_support import unsupported_local_file_message
from rag_core.ingest.local.models import LocalIngestPlan, LocalIngestRequest
from rag_core.ingest.sources.local import reject_local_hardlink_path, reject_local_symlink_path
from rag_core.manifest.persistence import (
    ManifestReconciliation,
    read_entries,
    reconcile_entries,
    validate_manifest_scope,
)
from rag_core.ingest.sources.local import LocalFileSourceReader, is_supported_local_candidate


def build_local_ingest_plan(request: LocalIngestRequest) -> LocalIngestPlan:
    validate_manifest_scope(request.namespace, request.collection)
    if request.max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")
    raw = str(request.path)
    source_plan = LocalFileSourceReader().read(raw)
    if not source_plan.items:
        candidate = Path(raw)
        if candidate.exists() and candidate.is_file():
            validate_supported_local_file(candidate, label="ingest path")
        raise ValueError(
            f"no supported files matched {raw!r} (use a literal path or a glob like './docs/**/*.md')"
        )
    return LocalIngestPlan(
        path=raw,
        namespace=request.namespace,
        collection=request.collection,
        documents=source_plan.items,
    )


def reconcile_local_ingest_plan(
    plan: LocalIngestPlan,
    *,
    manifest_dir: Path,
) -> ManifestReconciliation:
    entries = read_entries(
        manifest_dir,
        namespace=plan.namespace,
        collection=plan.collection,
    )
    return reconcile_entries(entries, plan.manifest_sources)


def validate_supported_local_file(path: Path, *, label: str = "path") -> None:
    if not path.exists():
        raise FileNotFoundError(str(path))
    if not path.is_file():
        raise ValueError(f"{label} must be a file: {str(path)!r}")
    reject_local_symlink_path(path)
    reject_local_hardlink_path(path)
    if not is_supported_local_candidate(path):
        raise ValueError(unsupported_local_file_message(path, label=label))


__all__ = [
    "build_local_ingest_plan",
    "reconcile_local_ingest_plan",
    "validate_supported_local_file",
]
