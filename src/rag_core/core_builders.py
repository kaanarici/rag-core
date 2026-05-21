"""Pure data builders for ingest and index wiring.

Single owner for the shallow transformations between ``PreparedDocument``,
``IngestedDocument`` and ``IndexRequest``.
"""

from __future__ import annotations

from rag_core.core_lifecycle import (
    compute_content_sha256,
    resolve_document_id,
    resolve_document_key,
)
from rag_core.core_models import (
    IngestedDocument,
    PreparedDocument,
    ProcessingFingerprint,
)
from rag_core.core_ocr_metadata import (
    OCR_METADATA_KEY as OCR_METADATA_KEY,
    read_ocr_metadata as read_ocr_metadata,
    write_ocr_metadata as write_ocr_metadata,
)
from rag_core.search.indexer_models import IndexRequest
from rag_core.search.policy import DEFAULT_POLICY, VectorStorePolicy
from rag_core.search.types import StoredDocumentRecord

def build_preview_document(
    *,
    file_bytes: bytes,
    prepared: PreparedDocument,
    namespace: str,
    corpus_id: str,
    document_id: str | None = None,
    document_key: str | None = None,
    metadata: dict[str, str] | None = None,
    collection_name: str | None = None,
    embedding_model: str | None = None,
    policy: VectorStorePolicy = DEFAULT_POLICY,
) -> IngestedDocument:
    resolved_document_key = resolve_document_key(
        filename=prepared.filename,
        path=prepared.path,
        document_key=document_key,
    )
    return IngestedDocument(
        document_id=resolve_document_id(
            namespace=namespace,
            corpus_id=corpus_id,
            document_key=resolved_document_key,
            document_id=document_id,
            policy=policy,
        ),
        corpus_id=corpus_id,
        namespace=namespace,
        chunk_count=len(prepared.chunks),
        filename=prepared.filename,
        mime_type=prepared.mime_type,
        document_key=resolved_document_key,
        content_sha256=compute_content_sha256(file_bytes),
        ingest_state="preview",
        replaced_existing=False,
        collection_name=collection_name,
        embedding_model=embedding_model,
        ocr=prepared.ocr,
        metadata={**prepared.metadata, **dict(metadata or {})},
    )


def build_index_request(
    *,
    prepared: PreparedDocument,
    document_id: str,
    document_key: str | None,
    content_sha256: str | None,
    processing_version: ProcessingFingerprint,
    existing: StoredDocumentRecord | None,
    corpus_id: str,
    namespace: str,
    source_type: str,
    metadata: dict[str, str] | None,
    embedding_model: str,
) -> IndexRequest:
    return IndexRequest(
        document_id=document_id,
        document_key=document_key,
        content_sha256=content_sha256,
        processing_version=processing_version.serialize(),
        existing_chunk_count=existing.chunk_count if existing is not None else None,
        corpus_id=corpus_id,
        namespace=namespace,
        text=prepared.markdown,
        filename=prepared.filename,
        mime_type=prepared.mime_type,
        source_type=source_type,
        path=_display_path_for_index(
            source_type=source_type,
            path=prepared.path,
            document_key=document_key,
        ),
        document_path=_stored_document_path(
            source_type=source_type,
            path=prepared.path,
        ),
        extra_fields=dict(metadata or {}) or None,
        embedding_model=embedding_model,
        pre_chunked_texts=[chunk.text for chunk in prepared.chunks],
        embedding_chunk_texts=[chunk.embedding_text for chunk in prepared.chunks],
        chunk_metadata=[dict(chunk.metadata) for chunk in prepared.chunks],
    )


def _display_path_for_index(
    *,
    source_type: str,
    path: str | None,
    document_key: str | None,
) -> str | None:
    if source_type in {"file", "archive"} and document_key:
        return document_key
    return path


def _stored_document_path(
    *,
    source_type: str,
    path: str | None,
) -> str | None:
    if source_type in {"file", "archive"}:
        return None
    return path


def build_ingested_document(
    *,
    prepared: PreparedDocument,
    document_id: str,
    corpus_id: str,
    namespace: str,
    document_key: str | None,
    content_sha256: str | None,
    chunk_count: int,
    ingest_state: str,
    replaced_existing: bool,
    collection_name: str,
    embedding_model: str,
    processing_version: ProcessingFingerprint,
    metadata: dict[str, str] | None,
) -> IngestedDocument:
    return IngestedDocument(
        document_id=document_id,
        corpus_id=corpus_id,
        namespace=namespace,
        chunk_count=chunk_count,
        filename=prepared.filename,
        mime_type=prepared.mime_type,
        document_key=document_key,
        content_sha256=content_sha256,
        ingest_state=ingest_state,
        replaced_existing=replaced_existing,
        collection_name=collection_name,
        embedding_model=embedding_model,
        processing_version=processing_version.serialize(),
        ocr=prepared.ocr,
        metadata={**prepared.metadata, **dict(metadata or {})},
    )
