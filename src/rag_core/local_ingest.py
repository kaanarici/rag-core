from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from rag_core.cli_provider_errors import (
    ProviderCliError,
    is_provider_bootstrap_error,
    provider_runtime_message,
)
from rag_core.config import INGEST_SOURCE_TYPE_FILE
from rag_core.core_file_io import detect_local_mime_type, read_file_bytes
from rag_core.core_lifecycle import compute_content_sha256
from rag_core.core_models import IngestedDocument
from rag_core.events.emit import emit_event, now_ms
from rag_core.events.types import (
    IngestBatchCompleted,
    IngestBatchFailed,
    IngestBatchStarted,
)
from rag_core.local_ingest_runner import (
    LocalIngestAborted,
    LocalIngestCore,
    LocalIngestCoreFactory,
    event_error_type as _event_error_type,
    ingest_local_documents as _ingest_local_documents,
    local_ingest_record_counts as _local_ingest_record_counts,
)
from rag_core.local_ingest_planning import (
    build_local_ingest_plan,
    reconcile_local_ingest_plan,
    validate_supported_local_file,
)
from rag_core.local_ingest_models import (
    LocalIngestFailure,
    LocalIngestPlan,
    LocalIngestRequest,
    LocalIngestResult,
    LocalIngestSuccess,
)
from rag_core.sources import LocalSourceItem
from rag_core.local_sources import (
    document_key as local_document_key,
    local_source_key_root,
    source_error_message,
)
from rag_core.manifest_preview_models import (
    ManifestPreviewRequest,
    ManifestPreviewResult,
)
from rag_core.manifest_preview_runner import preview_manifest
from rag_core.manifest_persistence import validate_manifest_scope
from rag_core.local_search_models import LocalSearchRequest, LocalSearchResult
from rag_core.local_search_runner import (
    LocalSearchCore,
    LocalSearchCoreFactory,
    default_corpus_id,
    discover_local_files,
    local_search_hit_payload,
    run_local_search,
)

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink


@runtime_checkable
class _IngestBytesCore(Protocol):
    async def ingest_bytes(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        namespace: str,
        corpus_id: str,
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
    core: _IngestBytesCore
    file_bytes: bytes
    expected_path: Path

    async def ensure_ready(self) -> None:
        return None

    async def ingest_file(
        self,
        file_path: Path,
        *,
        namespace: str,
        corpus_id: str,
        document_key: str,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
    ) -> IngestedDocument:
        if file_path != self.expected_path:
            raise ValueError("single-file ingest adapter received an unexpected path")
        return await self.core.ingest_bytes(
            file_bytes=self.file_bytes,
            filename=file_path.name,
            mime_type=detect_local_mime_type(file_path),
            namespace=namespace,
            corpus_id=corpus_id,
            document_key=document_key,
            path=str(file_path),
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
    started_ms = now_ms()
    emit_event(
        event_sink,
        IngestBatchStarted(
            namespace=plan.namespace,
            corpus_id=plan.corpus_id,
            planned_count=plan.document_count,
        ),
    )
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
            and isinstance(core, _IngestBytesCore)
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
            event_sink=event_sink,
            reconciliation=reconciliation,
        )
        return LocalIngestResult(
            namespace=plan.namespace,
            corpus_id=plan.corpus_id,
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
            succeeded_count, failed_count = _local_ingest_record_counts(records)
            emit_event(
                event_sink,
                IngestBatchFailed(
                    namespace=plan.namespace,
                    corpus_id=plan.corpus_id,
                    planned_count=plan.document_count,
                    completed_count=len(records),
                    succeeded_count=succeeded_count,
                    failed_count=failed_count,
                    duration_ms=now_ms() - started_ms,
                    error=_event_error_type(cli_error),
                ),
            )
            raise cli_error from cause
        raise cause from exc
    except Exception as exc:
        if ready and is_provider_bootstrap_error(exc):
            batch_failed = True
            cli_error = ProviderCliError(provider_runtime_message(exc, action="ingest"))
            succeeded_count, failed_count = _local_ingest_record_counts(records)
            emit_event(
                event_sink,
                IngestBatchFailed(
                    namespace=plan.namespace,
                    corpus_id=plan.corpus_id,
                    planned_count=plan.document_count,
                    completed_count=len(records),
                    succeeded_count=succeeded_count,
                    failed_count=failed_count,
                    duration_ms=now_ms() - started_ms,
                    error=_event_error_type(cli_error),
                ),
            )
            raise cli_error from exc
        batch_failed = True
        succeeded_count, failed_count = _local_ingest_record_counts(records)
        emit_event(
            event_sink,
            IngestBatchFailed(
                namespace=plan.namespace,
                corpus_id=plan.corpus_id,
                planned_count=plan.document_count,
                completed_count=len(records),
                succeeded_count=succeeded_count,
                failed_count=failed_count,
                duration_ms=now_ms() - started_ms,
                error=_event_error_type(exc),
            ),
        )
        raise
    finally:
        if not batch_failed:
            succeeded_count, failed_count = _local_ingest_record_counts(records)
            emit_event(
                event_sink,
                IngestBatchCompleted(
                    namespace=plan.namespace,
                    corpus_id=plan.corpus_id,
                    planned_count=plan.document_count,
                    succeeded_count=succeeded_count,
                    failed_count=failed_count,
                    duration_ms=now_ms() - started_ms,
                ),
            )
        if close_core and core is not None:
            await core.close()


async def _single_file_plan_with_bytes(
    request: LocalIngestRequest,
) -> _SingleFilePlanBytes | None:
    if request.max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")
    validate_manifest_scope(request.namespace, request.corpus_id)
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
            corpus_id=request.corpus_id,
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
    "LocalSearchCore",
    "LocalSearchCoreFactory",
    "LocalSearchRequest",
    "LocalSearchResult",
    "ManifestPreviewRequest",
    "ManifestPreviewResult",
    "build_local_ingest_plan",
    "default_corpus_id",
    "discover_local_files",
    "local_search_hit_payload",
    "preview_manifest",
    "reconcile_local_ingest_plan",
    "run_local_ingest",
    "run_local_ingest_with_core",
    "run_local_search",
    "validate_supported_local_file",
]
