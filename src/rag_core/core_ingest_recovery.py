"""Ingest recovery helpers for manifest writes, vector-index cleanup, and sidecar sync."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from posixpath import basename
from urllib.parse import urlsplit
from typing import TYPE_CHECKING, Any

from rag_core.core_ocr_metadata import read_ocr_metadata
from rag_core.core_manifest_builders import build_manifest_entry
from rag_core.core_models import CorpusManifestEntry, IngestedDocument
from rag_core.core_sidecar_sync import sync_search_sidecar
from rag_core.events.emit import emit_event, stage_guard
from rag_core.events.types import StageError
from rag_core.manifest_persistence import delete_entry, read_entries, write_entry
from rag_core.remote_document_keys import public_remote_document_key
from rag_core.search.indexer_models import IndexResult
from rag_core.search.policy import VectorStorePolicy
from rag_core.search.provider_protocols import SearchSidecar, VectorStore
from rag_core.search.request_models import StoredDocumentRecord

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink


def sync_sidecar_or_emit_error(
    *,
    sidecar: SearchSidecar | None,
    event_sink: "EventSink | None",
    namespace: str,
    corpus_id: str,
    document_id: str,
    result: IndexResult,
    policy: VectorStorePolicy,
) -> None:
    try:
        sync_search_sidecar(
            sidecar=sidecar,
            namespace=namespace,
            corpus_id=corpus_id,
            document_id=document_id,
            result=result,
            policy=policy,
        )
    except Exception as exc:
        emit_event(
            event_sink,
            StageError(stage="sidecar", error_type=type(exc).__name__),
        )
        raise


def write_final_manifest(
    *,
    event_sink: "EventSink | None",
    manifest_directory: Path,
    ingested: IngestedDocument,
) -> None:
    with stage_guard(event_sink, stage="manifest"):
        write_entry(
            manifest_directory,
            build_manifest_entry(ingested),
        )


def retry_final_manifest_write(
    *,
    event_sink: "EventSink | None",
    manifest_directory: Path,
    ingested: IngestedDocument,
) -> None:
    write_final_manifest(
        event_sink=event_sink,
        manifest_directory=manifest_directory,
        ingested=ingested,
    )


async def restore_manifest_from_vector_store(
    *,
    store: VectorStore,
    event_sink: "EventSink | None",
    manifest_directory: Path,
    namespace: str,
    corpus_id: str,
    document_id: str,
    filename: str,
    mime_type: str,
    document_key: str,
    fallback_entry: CorpusManifestEntry | None,
) -> None:
    """Rebuild the manifest entry from the vector store after ingest rollback."""
    if store.capabilities.document_record_lookup:
        record = await store.get_document_record(
            namespace=namespace,
            corpus_id=corpus_id,
            document_id=document_id,
        )
        with stage_guard(event_sink, stage="manifest"):
            if record is None:
                delete_entry(
                    manifest_directory,
                    namespace=namespace,
                    corpus_id=corpus_id,
                    document_id=document_id,
                )
                return
            write_entry(
                manifest_directory,
                restored_manifest_entry_from_existing_record(
                    record,
                    previous=fallback_entry,
                    filename=filename,
                    mime_type=mime_type,
                    document_key=document_key,
                ),
            )
        return
    if fallback_entry is not None:
        with stage_guard(event_sink, stage="manifest"):
            write_entry(manifest_directory, fallback_entry)


def latest_manifest_entry(
    directory: Path,
    *,
    namespace: str,
    corpus_id: str,
    document_id: str,
) -> CorpusManifestEntry | None:
    for entry in read_entries(directory, namespace=namespace, corpus_id=corpus_id):
        if entry.document_id == document_id:
            return entry
    return None


def manifest_entry_from_existing_record(
    record: StoredDocumentRecord,
    *,
    filename: str,
    mime_type: str,
    document_key: str | None = None,
) -> CorpusManifestEntry:
    return CorpusManifestEntry(
        document_id=record.document_id,
        namespace=record.namespace,
        corpus_id=record.corpus_id,
        document_key=_existing_document_key(
            record,
            previous=None,
            candidate=document_key,
        ),
        content_sha256=record.content_sha256,
        filename=_existing_filename(record, fallback=filename),
        mime_type=mime_type,
        chunk_count=record.chunk_count,
    )


def restored_manifest_entry_from_existing_record(
    record: StoredDocumentRecord,
    *,
    previous: CorpusManifestEntry | None,
    filename: str,
    mime_type: str,
    document_key: str | None = None,
) -> CorpusManifestEntry:
    if previous is None:
        return manifest_entry_from_existing_record(
            record,
            filename=filename,
            mime_type=mime_type,
            document_key=document_key,
        )
    return replace(
        previous,
        document_id=record.document_id,
        namespace=record.namespace,
        corpus_id=record.corpus_id,
        document_key=_existing_document_key(
            record,
            previous=previous.document_key,
            candidate=document_key,
        ),
        content_sha256=record.content_sha256,
        chunk_count=record.chunk_count,
    )


def refreshed_manifest_entry(
    *,
    previous: CorpusManifestEntry | None,
    existing: StoredDocumentRecord | None,
    document_id: str,
    namespace: str,
    corpus_id: str,
    document_key: str | None,
    content_sha256: str,
    filename: str,
    mime_type: str,
    metadata: dict[str, Any] | None,
) -> CorpusManifestEntry:
    base = previous
    if base is None and existing is not None:
        base = manifest_entry_from_existing_record(
            existing,
            filename=filename,
            mime_type=mime_type,
            document_key=document_key,
        )
    merged_metadata = dict(base.metadata) if base is not None else {}
    if metadata:
        merged_metadata.update(metadata)
    parser = manifest_parser(merged_metadata.get("parser"))
    if parser is None and base is not None:
        parser = base.parser
    ocr = read_ocr_metadata(merged_metadata)
    needs_ocr = bool(merged_metadata.get("needs_ocr")) or ocr.provider is not None
    if base is not None:
        needs_ocr = needs_ocr or base.needs_ocr
    if base is not None:
        return replace(
            base,
            document_id=document_id,
            namespace=namespace,
            corpus_id=corpus_id,
            document_key=(
                _existing_document_key(
                    existing,
                    previous=base.document_key,
                    candidate=document_key,
                )
                if existing is not None
                else _resolved_document_key(
                    preferred=document_key,
                    previous=base.document_key,
                )
            ),
            content_sha256=_refreshed_content_sha256(existing, content_sha256),
            filename=(
                _existing_filename(existing, fallback=base.filename)
                if existing is not None
                else filename
            ),
            mime_type=mime_type,
            chunk_count=existing.chunk_count if existing is not None else base.chunk_count,
            parser=parser,
            needs_ocr=needs_ocr,
            metadata=merged_metadata,
        )
    return CorpusManifestEntry(
        document_id=document_id,
        namespace=namespace,
        corpus_id=corpus_id,
        document_key=(
            _existing_document_key(existing, previous=None, candidate=document_key)
            if existing is not None
            else document_key
        ),
        content_sha256=content_sha256,
        filename=filename,
        mime_type=mime_type,
        chunk_count=0,
        parser=parser,
        needs_ocr=needs_ocr,
        metadata=merged_metadata,
    )


def _refreshed_content_sha256(
    existing: StoredDocumentRecord | None,
    fresh_content_sha256: str,
) -> str:
    if existing is not None and existing.content_sha256:
        return existing.content_sha256
    return fresh_content_sha256


def _resolved_document_key(
    *,
    preferred: str | None,
    fallback: str | None = None,
    previous: str | None = None,
) -> str | None:
    if preferred is not None and preferred.strip():
        return preferred
    if fallback is not None and fallback.strip():
        return fallback
    if previous is not None and previous.strip():
        return previous
    return None


def _existing_document_key(
    record: StoredDocumentRecord,
    *,
    previous: str | None,
    candidate: str | None,
) -> str | None:
    return _resolved_document_key(
        preferred=record.document_key,
        previous=previous,
        fallback=candidate,
    )


def _existing_filename(record: StoredDocumentRecord, *, fallback: str) -> str:
    return filename_from_document_key(record.document_key, fallback=fallback)


def filename_from_document_key(document_key: str | None, *, fallback: str) -> str:
    if document_key is None or not document_key.strip():
        return fallback
    normalized = _public_document_key(document_key.strip())
    if normalized.startswith(("local:", "archive:")):
        normalized = normalized.split(":", 1)[1]
    elif normalized.startswith("url:"):
        return _filename_from_url_document_key(normalized, fallback=fallback)
    normalized = normalized.split("#source:", 1)[0]
    name = basename(normalized.rstrip("/"))
    return name or fallback


def _filename_from_url_document_key(document_key: str, *, fallback: str) -> str:
    raw_url = document_key.removeprefix("url:")
    if "#path:" in raw_url:
        return fallback
    parsed = urlsplit(raw_url)
    name = basename(parsed.path.rstrip("/"))
    return name or fallback


def _public_document_key(document_key: str) -> str:
    return public_remote_document_key(document_key)


def manifest_parser(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    parser = value.strip()
    return parser or None


__all__ = [
    "filename_from_document_key",
    "latest_manifest_entry",
    "manifest_entry_from_existing_record",
    "manifest_parser",
    "refreshed_manifest_entry",
    "restored_manifest_entry_from_existing_record",
    "restore_manifest_from_vector_store",
    "retry_final_manifest_write",
    "sync_sidecar_or_emit_error",
    "write_final_manifest",
]
