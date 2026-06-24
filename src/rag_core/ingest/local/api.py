from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from rag_core.provider_errors import (
    ProviderCliError,
    is_provider_bootstrap_error,
    provider_runtime_message,
)
from rag_core.config import INGEST_SOURCE_TYPE_FILE
from rag_core.file_io import detect_local_mime_type, read_file_bytes
from rag_core.file_io import compute_content_sha256
from rag_core.core_models import IngestedDocument
from rag_core.ingest.lifecycle import (
    IngestBatchLifecycle,
)
from rag_core.ingest.local.runner import (
    LocalIngestAborted,
    LocalIngestCore,
    LocalIngestCoreFactory,
    event_error_type as _event_error_type,
    ingest_local_documents as _ingest_local_documents,
)
from rag_core.ingest.local.planning import (
    build_local_ingest_plan,
    reconcile_local_ingest_plan,
    validate_supported_local_file,
)
from rag_core.ingest.local.models import (
    LocalIngestFailure,
    LocalIngestPlan,
    LocalIngestRequest,
    LocalIngestResult,
    LocalIngestSuccess,
)
from rag_core.ingest.sources.local import (
    LocalSourceItem,
    document_key as local_document_key,
    local_source_key_root,
    source_error_message,
)
from rag_core.manifest.preview.models import (
    ManifestPreviewRequest,
    ManifestPreviewResult,
)
from rag_core.manifest.preview.runner import preview_manifest
from rag_core.manifest.persistence import validate_manifest_scope
from rag_core.local_search.models import LocalSearchRequest, LocalSearchResult
from rag_core.local_search.runner import (
    LocalContextCore,
    LocalContextCoreFactory,
    LocalSearchCore,
    LocalSearchCoreFactory,
    default_collection,
    discover_local_files,
    local_search_hit_payload,
    run_local_context,
    run_local_search,
)

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink


@runtime_checkable
class _AddBytesCore(Protocol):
    async def add_bytes(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        namespace: str,
        collection: str,
        document_id: str | None = None,
        document_key: str | None = None,
        path: str | None = None,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        source_type: str | None = None,
    ) -> IngestedDocument: ...


@dataclass(frozen=True)
class _SingleFilePlanBytes:
    plan: LocalIngestPlan
    file_bytes: bytes | None


@dataclass(frozen=True)
class _BytesIngestAdapter:
    core: _AddBytesCore
    file_bytes: bytes
    expected_path: Path

    async def ensure_ready(self) -> None:
        return None

    async def add_file(
        self,
        path: Path,
        *,
        namespace: str,
        collection: str,
        document_key: str,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        pre_read_bytes: bytes | None = None,
    ) -> IngestedDocument:
        if path != self.expected_path:
            raise ValueError("single-file ingest adapter received an unexpected path")
        # Prefer the runner's fresh ingest-time read over the plan-time bytes:
        # if the file changed between discovery and ingest, index what is on
        # disk now. The plan-time bytes are only a fallback when the runner did
        # not pre-read. Matches the canonical pattern in facade.ingest_sources.
        return await self.core.add_bytes(
            file_bytes=pre_read_bytes if pre_read_bytes is not None else self.file_bytes,
            filename=path.name,
            mime_type=detect_local_mime_type(path),
            namespace=namespace,
            collection=collection,
            document_key=document_key,
            path=str(path),
            metadata=metadata,
            force_reindex=force_reindex,
            source_type=INGEST_SOURCE_TYPE_FILE,
        )

    async def close(self) -> None:
        return None


async def run_local_ingest(
    request: LocalIngestRequest,
    *,
    core_factory: LocalIngestCoreFactory,
    event_sink: "EventSink | None" = None,
    manifest_dir: Path | None = None,
) -> LocalIngestResult:
    return await _run_local_ingest(
        request,
        core_factory=core_factory,
        close_core=True,
        event_sink=event_sink,
        manifest_dir=manifest_dir,
    )


async def run_local_ingest_with_core(
    request: LocalIngestRequest,
    *,
    core: LocalIngestCore,
    event_sink: "EventSink | None" = None,
    manifest_dir: Path | None = None,
) -> LocalIngestResult:
    return await _run_local_ingest(
        request,
        core_factory=lambda: core,
        close_core=False,
        event_sink=event_sink,
        manifest_dir=manifest_dir,
    )


async def _run_local_ingest(
    request: LocalIngestRequest,
    *,
    core_factory: LocalIngestCoreFactory,
    close_core: bool,
    event_sink: "EventSink | None" = None,
    manifest_dir: Path | None = None,
) -> LocalIngestResult:
    single_file_plan = await _single_file_plan_with_bytes(request)
    plan = (
        single_file_plan.plan
        if single_file_plan is not None
        else build_local_ingest_plan(request)
    )
    reconciliation = (
        reconcile_local_ingest_plan(plan, manifest_dir=manifest_dir)
        if manifest_dir is not None
        else None
    )
    lifecycle = IngestBatchLifecycle[LocalIngestSuccess | LocalIngestFailure](
        event_sink=event_sink,
        namespace=plan.namespace,
        collection=plan.collection,
        planned_count=plan.document_count,
        is_success=lambda record: isinstance(record, LocalIngestSuccess),
        error_type=_event_error_type,
    )
    lifecycle.started()
    core: LocalIngestCore | None = None
    records: list[LocalIngestSuccess | LocalIngestFailure] = []
    batch_failed = False
    ready = False
    try:
        core = core_factory()
        await core.ensure_ready()
        ready = True
        ingest_core: LocalIngestCore = core
        if (
            single_file_plan is not None
            and single_file_plan.file_bytes is not None
            and isinstance(core, _AddBytesCore)
        ):
            ingest_core = _BytesIngestAdapter(
                core=core,
                file_bytes=single_file_plan.file_bytes,
                expected_path=plan.documents[0].path,
            )
        records = await _ingest_local_documents(
            ingest_core,
            plan,
            request,
            lifecycle=lifecycle,
            reconciliation=reconciliation,
        )
        return LocalIngestResult(
            namespace=plan.namespace,
            collection=plan.collection,
            records=tuple(records),
        )
    except LocalIngestAborted as exc:
        records = list(exc.records)
        cause = exc.cause
        if ready and is_provider_bootstrap_error(cause):
            batch_failed = True
            cli_error = ProviderCliError(
                provider_runtime_message(cause, action="ingest")
            )
            lifecycle.failed(error=cli_error, records=records)
            raise cli_error from cause
        raise cause from exc
    except Exception as exc:
        if ready and is_provider_bootstrap_error(exc):
            batch_failed = True
            cli_error = ProviderCliError(provider_runtime_message(exc, action="ingest"))
            lifecycle.failed(error=cli_error, records=records)
            raise cli_error from exc
        batch_failed = True
        lifecycle.failed(error=exc, records=records)
        raise
    finally:
        if not batch_failed:
            lifecycle.completed(records)
        if close_core and core is not None:
            await core.close()


async def _single_file_plan_with_bytes(
    request: LocalIngestRequest,
) -> _SingleFilePlanBytes | None:
    if request.max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")
    validate_manifest_scope(request.namespace, request.collection)
    raw = str(request.path)
    if any(char in raw for char in "*?["):
        return None
    path = Path(raw)
    if not path.exists() or not path.is_file():
        return None
    validate_supported_local_file(path, label="ingest path")
    root = local_source_key_root(raw)
    try:
        read_bytes = await read_file_bytes(path)
    except OSError as exc:
        file_bytes = None
        content_sha256 = None
        source_error = source_error_message(exc)
    else:
        file_bytes = read_bytes
        content_sha256 = compute_content_sha256(file_bytes)
        source_error = ""
    return _SingleFilePlanBytes(
        plan=LocalIngestPlan(
            path=raw,
            namespace=request.namespace,
            collection=request.collection,
            documents=(
                LocalSourceItem(
                    path=path,
                    document_key=local_document_key(root, path),
                    content_sha256=content_sha256,
                    source_error=source_error,
                ),
            ),
        ),
        file_bytes=file_bytes,
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
