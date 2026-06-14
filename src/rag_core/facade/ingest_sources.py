from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from rag_core.config import INGEST_SOURCE_TYPE_FILE
from rag_core._engine.core_file_io import detect_local_mime_type, read_file_bytes
from rag_core.core_models import IngestedDocument, RAGCoreConfig
from rag_core._engine.core_remote import IngestBytes, ingest_remote_url
from rag_core.local_sources import document_key as local_document_key
from rag_core.local_sources import local_source_key_root
from rag_core.local_sources import reject_local_hardlink_path
from rag_core.local_sources import reject_local_symlink_path

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink
    from rag_core.events.types import AuditContext
    from rag_core.fetch_security import FetchLimits, FetchSecurityPolicy
    from rag_core.fetching import FetchClient


class LocalIngestBytes(Protocol):
    async def __call__(
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
        audit_context: "AuditContext | None" = None,
        ingest_id: str | None = None,
    ) -> IngestedDocument: ...


async def ingest_local_file_source(
    path: str | Path,
    *,
    ingest_bytes: LocalIngestBytes,
    namespace: str,
    corpus_id: str,
    document_id: str | None = None,
    document_key: str | None = None,
    metadata: dict[str, str] | None = None,
    force_reindex: bool = False,
    audit_context: "AuditContext | None" = None,
    ingest_id: str | None = None,
    pre_read_bytes: bytes | None = None,
) -> IngestedDocument:
    file_path = Path(path)
    reject_local_symlink_path(file_path)
    reject_local_hardlink_path(file_path)
    resolved_document_key = _resolve_local_document_key(
        path=path,
        file_path=file_path,
        document_key=document_key,
    )
    file_bytes = pre_read_bytes if pre_read_bytes is not None else await read_file_bytes(file_path)
    return await ingest_bytes(
        file_bytes=file_bytes,
        filename=file_path.name,
        mime_type=detect_local_mime_type(file_path),
        namespace=namespace,
        corpus_id=corpus_id,
        document_id=document_id,
        document_key=resolved_document_key,
        path=str(file_path),
        metadata=metadata,
        force_reindex=force_reindex,
        source_type=INGEST_SOURCE_TYPE_FILE,
        audit_context=audit_context,
        ingest_id=ingest_id,
    )


def _resolve_local_document_key(
    *,
    path: str | Path,
    file_path: Path,
    document_key: str | None,
) -> str:
    if document_key and document_key.strip():
        return document_key.strip()
    if _contains_parent_traversal(path):
        raise ValueError(
            "document_key is required for single-file ingest when path traversal segments prevent inferring a stable collection root"
        )
    return local_document_key(local_source_key_root(str(path)), file_path)


def _contains_parent_traversal(path: str | Path) -> bool:
    raw = Path(path)
    return any(part == ".." for part in raw.parts)


def resolve_archive_manifest_dir(
    *,
    config: RAGCoreConfig,
    manifest_dir: str | Path | None,
) -> str | Path | None:
    if manifest_dir is not None:
        return manifest_dir
    return config.ingest.manifest_directory


async def ingest_remote_url_source(
    url: str,
    *,
    ingest_bytes: IngestBytes,
    namespace: str,
    corpus_id: str,
    event_sink: "EventSink | None" = None,
    document_id: str | None = None,
    metadata: dict[str, str] | None = None,
    force_reindex: bool = False,
    fetch_client: "FetchClient | None" = None,
    fetch_policy: "FetchSecurityPolicy | None" = None,
    fetch_limits: "FetchLimits | None" = None,
) -> IngestedDocument:
    return await ingest_remote_url(
        url,
        ingest_bytes=ingest_bytes,
        namespace=namespace,
        corpus_id=corpus_id,
        event_sink=event_sink,
        document_id=document_id,
        metadata=metadata,
        force_reindex=force_reindex,
        fetch_client=fetch_client,
        fetch_policy=fetch_policy,
        fetch_limits=fetch_limits,
    )
