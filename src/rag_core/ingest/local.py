"""Folder ingest: scan once, read each file once, and use the engine's document path."""

from __future__ import annotations

from rag_core._async import run_sync
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, TYPE_CHECKING
from rag_core.config.ingest_config import DEFAULT_INGEST_MAX_CONCURRENCY
from rag_core.core_models import IngestedDocument
from rag_core.core_models import DeleteDocumentResult
from rag_core.documents.formats import unsupported_local_file_message
from rag_core.documents.exception_names import exception_type
from rag_core.file_io import compute_content_sha256, read_file_bytes
from rag_core.ingest.batch import BatchEvents, bounded_map
from rag_core.ingest.sources.local import (
    LocalSourceItem,
    read_local_source,
    local_source_id,
    reject_local_symlink_path,
    reject_local_hardlink_path,
    is_supported_local_candidate,
)
from rag_core.manifest.persistence import read_entries, validate_manifest_scope
from rag_core.provider_errors import (
    is_provider_bootstrap_error,
    ProviderCliError,
    provider_runtime_message,
)
from rag_core.safe_messages import safe_error_message

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink


class LocalIngestCore(Protocol):
    async def ensure_ready(self) -> None: ...
    async def add_file(
        self,
        path: Path,
        *,
        tenant_id: str,
        collection: str,
        document_key: str,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        pre_read_bytes: bytes | None = None,
    ) -> IngestedDocument: ...
    async def delete_document(
        self, *, document_id: str, tenant_id: str, collection: str
    ) -> DeleteDocumentResult: ...


@dataclass(frozen=True)
class LocalIngestRequest:
    path: str | Path
    tenant_id: str
    collection: str
    metadata: dict[str, str] | None = None
    force_reindex: bool = False
    max_concurrency: int = DEFAULT_INGEST_MAX_CONCURRENCY
    max_files: int | None = None
    prune_missing: bool = False


@dataclass(frozen=True)
class LocalIngestPlan:
    path: str
    tenant_id: str
    collection: str
    documents: tuple[LocalSourceItem, ...]
    truncated: bool = False

    @property
    def document_count(self) -> int:
        return len(self.documents)

    def to_payload(self) -> dict[str, object]:
        return {
            "path": "<local-source>",
            "tenant_id": self.tenant_id,
            "collection": self.collection,
            "planned_count": self.document_count,
            "truncated": self.truncated,
            "documents": [
                {
                    "path": "<local-file>",
                    "filename": document.path.name,
                    "source_error": "source read failed"
                    if document.source_error
                    else "",
                }
                for document in self.documents
            ],
        }


@dataclass(frozen=True)
class LocalIngestSuccess:
    path: str
    document_key: str
    content_sha256: str | None
    document_id: str
    filename: str
    chunk_count: int
    ingest_state: str
    replaced_existing: bool

    def to_payload(self, *, include_private: bool = False) -> dict[str, object]:
        if include_private:
            return {"ok": True, **asdict(self)}
        return {
            "ok": True,
            "path": _public_source_path(self.path),
            "filename": self.filename,
            "document_id": self.document_id,
            "chunk_count": self.chunk_count,
            "ingest_state": self.ingest_state,
            "replaced_existing": self.replaced_existing,
        }


@dataclass(frozen=True)
class LocalIngestFailure:
    path: str
    document_key: str
    content_sha256: str | None
    error: str

    def to_payload(self, *, include_private: bool = False) -> dict[str, object]:
        if include_private:
            return {"ok": False, **asdict(self)}
        return {
            "ok": False,
            "path": _public_source_path(self.path),
            "filename": Path(self.path).name,
            "error": self.error,
        }


def _public_source_path(path: str) -> str:
    archive_path, separator, member_path = path.partition("!/")
    if separator:
        return f"{Path(archive_path).name}!/{member_path}"
    return "<local-file>"


@dataclass(frozen=True)
class LocalIngestResult:
    tenant_id: str
    collection: str
    records: tuple[LocalIngestSuccess | LocalIngestFailure, ...]
    truncated: bool = False

    @property
    def planned_count(self) -> int:
        return len(self.records)

    @property
    def succeeded(self) -> tuple[LocalIngestSuccess, ...]:
        return tuple(r for r in self.records if isinstance(r, LocalIngestSuccess))

    @property
    def written(self) -> tuple[LocalIngestSuccess, ...]:
        return tuple(r for r in self.succeeded if r.ingest_state != "unchanged")

    @property
    def skipped(self) -> tuple[LocalIngestSuccess, ...]:
        return tuple(r for r in self.succeeded if r.ingest_state == "unchanged")

    @property
    def failed(self) -> tuple[LocalIngestFailure, ...]:
        return tuple(r for r in self.records if isinstance(r, LocalIngestFailure))

    @property
    def succeeded_count(self) -> int:
        return sum(isinstance(r, LocalIngestSuccess) for r in self.records)

    @property
    def failed_count(self) -> int:
        return len(self.records) - self.succeeded_count

    @property
    def written_count(self) -> int:
        return len(self.written)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)

    def to_payload(self, *, include_private: bool = False) -> dict[str, object]:
        return {
            "tenant_id": self.tenant_id,
            "collection": self.collection,
            "planned_count": self.planned_count,
            "succeeded_count": self.succeeded_count,
            "written_count": self.written_count,
            "skipped_count": self.skipped_count,
            "failed_count": self.failed_count,
            "truncated": self.truncated,
            "records": [
                r.to_payload(include_private=include_private) for r in self.records
            ],
        }


def build_local_ingest_plan(
    request: LocalIngestRequest, *, hash_content: bool = True
) -> LocalIngestPlan:
    validate_manifest_scope(request.tenant_id, request.collection)
    if request.max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")
    if request.max_files is not None and request.max_files <= 0:
        raise ValueError("max_files must be positive")
    raw = str(request.path)
    source_plan = read_local_source(raw, hash_content=hash_content)
    if not source_plan.items and not (request.prune_missing and Path(raw).is_dir()):
        candidate = Path(raw)
        if candidate.exists() and candidate.is_file():
            validate_supported_local_file(candidate, label="ingest path")
        raise ValueError(
            f"no supported files matched {raw!r} (use a literal path or a glob like './docs/**/*.md')"
        )
    documents = source_plan.items
    truncated = False
    if request.max_files is not None:
        truncated = len(documents) > request.max_files
        documents = documents[: request.max_files]
    return LocalIngestPlan(
        path=raw,
        tenant_id=request.tenant_id,
        collection=request.collection,
        documents=documents,
        truncated=truncated,
    )


def validate_supported_local_file(path: Path, *, label: str = "path") -> None:
    if not path.exists():
        raise FileNotFoundError(str(path))
    if not path.is_file():
        raise ValueError(f"{label} must be a file: {str(path)!r}")
    reject_local_symlink_path(path)
    reject_local_hardlink_path(path)
    if not is_supported_local_candidate(path):
        raise ValueError(unsupported_local_file_message(path, label=label))


async def run_local_ingest_with_core(
    request: LocalIngestRequest,
    *,
    core: LocalIngestCore,
    event_sink: EventSink | None = None,
    manifest_dir: Path | None = None,
    max_document_bytes: int | None = None,
) -> LocalIngestResult:
    if request.prune_missing and (
        manifest_dir is None or not Path(request.path).is_dir()
    ):
        raise ValueError(
            "prune_missing requires a literal folder and persistent manifest directory"
        )
    plan = await run_sync(build_local_ingest_plan, request, hash_content=False)
    previous = (
        read_entries(manifest_dir, request.tenant_id, request.collection)
        if request.prune_missing and manifest_dir is not None and not plan.truncated
        else []
    )
    batch = BatchEvents(
        event_sink=event_sink,
        tenant_id=plan.tenant_id,
        collection=plan.collection,
        planned_count=plan.document_count,
    )
    batch.started()
    ready = False
    try:
        await core.ensure_ready()
        ready = True

        async def ingest(position: int, document: LocalSourceItem) -> None:
            error = ""
            digest = document.content_sha256
            try:
                if document.source_error:
                    raise OSError("source read failed")
                data = await read_file_bytes(document.path, max_bytes=max_document_bytes)
                digest = compute_content_sha256(data)
                item = await core.add_file(
                    document.path,
                    tenant_id=plan.tenant_id,
                    collection=plan.collection,
                    document_key=document.document_key,
                    metadata=request.metadata,
                    force_reindex=request.force_reindex,
                    pre_read_bytes=data,
                )
                digest = item.content_sha256 or digest
                record: LocalIngestSuccess | LocalIngestFailure = LocalIngestSuccess(
                    path=str(document.path),
                    document_key=document.document_key,
                    content_sha256=digest,
                    document_id=item.document_id,
                    filename=item.filename,
                    chunk_count=item.chunk_count,
                    ingest_state=item.ingest_state,
                    replaced_existing=item.replaced_existing,
                )
            except Exception as exc:
                if is_provider_bootstrap_error(exc):
                    raise
                error = exception_type(exc)
                record = LocalIngestFailure(
                    path=str(document.path),
                    document_key=document.document_key,
                    content_sha256=digest,
                    error=safe_error_message(exc, action="ingest"),
                )
            batch.record(position, record, error=error)

        await bounded_map(
            plan.documents, ingest, max_concurrency=request.max_concurrency
        )
        if previous:
            discovered = {document.document_key for document in plan.documents}
            suffix = f"#source:{local_source_id(Path(request.path))}"
            for entry in previous:
                key = entry.document_key or ""
                if (
                    key not in discovered
                    and key.startswith("local:")
                    and key.endswith(suffix)
                ):
                    await core.delete_document(
                        document_id=entry.document_id,
                        tenant_id=plan.tenant_id,
                        collection=plan.collection,
                    )
        batch.completed()
    except BaseException as exc:
        if ready and isinstance(exc, Exception) and is_provider_bootstrap_error(exc):
            error = ProviderCliError(provider_runtime_message(exc, action="ingest"))
            batch.failed(error)
            raise error from None
        batch.failed(exc)
        raise
    return LocalIngestResult(
        tenant_id=plan.tenant_id,
        collection=plan.collection,
        records=tuple(batch.records),
        truncated=plan.truncated,
    )
