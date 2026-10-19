"""Local ingest records and recovery; indexed chunks remain the source of truth."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from posixpath import basename
from urllib.parse import urlsplit
from typing import TYPE_CHECKING, Any
from rag_core._engine.core_ocr_metadata import read_ocr_metadata
from rag_core.core_models import CollectionManifestEntry, IngestedDocument
from rag_core.events.emit import stage_guard
from rag_core.ingest.remote_keys import public_remote_document_key
from rag_core.manifest.persistence import manifest_path, write_entry
from rag_core.manifest.jsonl import latest_manifest_jsonl_entry
from rag_core.manifest.entries import sanitize_manifest_metadata
from rag_core.search.request_models import StoredDocumentRecord

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink


def write_final_manifest(
    *,
    event_sink: EventSink | None,
    manifest_directory: Path,
    ingested: IngestedDocument,
) -> None:
    with stage_guard(event_sink, stage="manifest"):
        write_entry(manifest_directory, build_manifest_entry(ingested))


def latest_manifest_entry(
    directory: Path, *, tenant_id: str, collection: str, document_id: str
) -> CollectionManifestEntry | None:
    entry = latest_manifest_jsonl_entry(
        manifest_path(directory, tenant_id, collection), document_id
    )
    if entry is not None and (entry.tenant_id, entry.collection) != (
        tenant_id,
        collection,
    ):
        from rag_core.manifest.entries import ManifestReadError

        raise ManifestReadError(
            manifest_path(directory, tenant_id, collection), 0, "invalid_entry"
        )
    return entry


def _document_key(*candidates: str | None) -> str | None:
    return next((key for key in candidates if key and key.strip()), None)


def manifest_entry_from_existing_record(
    record: StoredDocumentRecord,
    *,
    filename: str,
    mime_type: str,
    document_key: str | None = None,
    previous: CollectionManifestEntry | None = None,
) -> CollectionManifestEntry:
    base = previous or CollectionManifestEntry(
        document_id=record.document_id,
        tenant_id=record.tenant_id,
        collection=record.collection,
        document_key=document_key,
        content_sha256=record.content_sha256,
        filename=filename_from_document_key(record.document_key, fallback=filename),
        mime_type=mime_type,
        chunk_count=record.chunk_count,
    )
    return replace(
        base,
        document_id=record.document_id,
        tenant_id=record.tenant_id,
        collection=record.collection,
        document_key=_document_key(
            record.document_key, document_key, base.document_key
        ),
        content_sha256=record.content_sha256,
        chunk_count=record.chunk_count,
    )


def refreshed_manifest_entry(
    *,
    previous: CollectionManifestEntry | None,
    existing: StoredDocumentRecord | None,
    document_id: str,
    tenant_id: str,
    collection: str,
    document_key: str | None,
    content_sha256: str,
    filename: str,
    mime_type: str,
    metadata: dict[str, Any] | None,
) -> CollectionManifestEntry:
    base = previous or CollectionManifestEntry(
        document_id=document_id,
        tenant_id=tenant_id,
        collection=collection,
        document_key=document_key,
        content_sha256=content_sha256,
        filename=filename,
        mime_type=mime_type,
        chunk_count=0,
    )
    merged = {**base.metadata, **(metadata or {})}
    parser = merged.get("parser")
    return replace(
        base,
        document_id=document_id,
        tenant_id=tenant_id,
        collection=collection,
        document_key=_document_key(
            existing.document_key if existing else None, document_key, base.document_key
        ),
        content_sha256=(existing.content_sha256 if existing else None)
        or content_sha256,
        filename=filename_from_document_key(
            existing.document_key, fallback=base.filename
        )
        if existing
        else filename,
        mime_type=mime_type,
        chunk_count=existing.chunk_count if existing else base.chunk_count,
        parser=(parser.strip() if isinstance(parser, str) else "") or base.parser,
        needs_ocr=bool(merged.get("needs_ocr"))
        or read_ocr_metadata(merged).provider is not None
        or base.needs_ocr,
        metadata=merged,
    )


def filename_from_document_key(document_key: str | None, *, fallback: str) -> str:
    if not document_key or not document_key.strip():
        return fallback
    normalized = public_remote_document_key(document_key.strip())
    if normalized.startswith(("local:", "archive:")):
        normalized = normalized.split(":", 1)[1]
    elif normalized.startswith("url:"):
        raw = normalized.removeprefix("url:")
        return (
            fallback
            if "#path:" in raw
            else basename(urlsplit(raw).path.rstrip("/")) or fallback
        )
    return basename(normalized.split("#source:", 1)[0].rstrip("/")) or fallback


def build_manifest_entry(document: IngestedDocument) -> CollectionManifestEntry:
    ocr = read_ocr_metadata(document.metadata)
    return CollectionManifestEntry(
        document_id=document.document_id,
        tenant_id=document.tenant_id,
        collection=document.collection,
        document_key=document.document_key,
        content_sha256=document.content_sha256,
        filename=document.filename,
        mime_type=document.mime_type,
        chunk_count=document.chunk_count,
        parser=_optional_str(document.metadata.get("parser")),
        needs_ocr=bool(document.metadata.get("needs_ocr") or document.ocr.needed)
        or ocr.provider is not None,
        metadata=sanitize_manifest_metadata(document.metadata),
    )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
