from __future__ import annotations

from dataclasses import replace

from rag_core._engine.core_builders import build_ingested_document
from rag_core._engine.core_ingest_decision import IngestDecision
from rag_core._engine.core_ingest_identity import ResolvedIngestIdentity
from rag_core._engine.core_ingest_recovery import filename_from_document_key
from rag_core.core_models import IngestedDocument, PreparedDocument
from rag_core.search.indexer_models import IndexResult


def build_skipped_ingested_document(
    *,
    prepared: PreparedDocument,
    identity: ResolvedIngestIdentity,
    decision: IngestDecision,
    collection: str,
    namespace: str,
    collection_name: str,
    embedding_model: str,
    metadata: dict[str, str] | None,
) -> IngestedDocument:
    existing = decision.existing
    document_key = identity.document_key
    content_sha256 = identity.content_sha256
    chunk_count = 0
    if existing is not None:
        document_key = existing.document_key or document_key
        content_sha256 = existing.content_sha256 or content_sha256
        chunk_count = existing.chunk_count
        prepared = _prepared_with_existing_filename(prepared, existing.document_key)
    return build_ingested_document(
        prepared=prepared,
        document_id=identity.document_id,
        collection=collection,
        namespace=namespace,
        document_key=document_key,
        content_sha256=content_sha256,
        chunk_count=chunk_count,
        ingest_state=decision.ingest_state,
        replaced_existing=False,
        collection_name=collection_name,
        embedding_model=embedding_model,
        processing_version=identity.processing_version,
        metadata=metadata,
    )


def build_fast_skipped_ingested_document(
    *,
    identity: ResolvedIngestIdentity,
    decision: IngestDecision,
    filename: str,
    mime_type: str,
    collection: str,
    namespace: str,
    collection_name: str,
    embedding_model: str,
    metadata: dict[str, str] | None,
) -> IngestedDocument:
    existing = decision.existing
    document_key = identity.document_key
    content_sha256 = identity.content_sha256
    chunk_count = 0
    if existing is not None:
        document_key = existing.document_key or document_key
        content_sha256 = existing.content_sha256 or content_sha256
        chunk_count = existing.chunk_count
        filename = filename_from_document_key(existing.document_key, fallback=filename)
    return IngestedDocument(
        document_id=identity.document_id,
        collection=collection,
        namespace=namespace,
        chunk_count=chunk_count,
        filename=filename,
        mime_type=mime_type,
        document_key=document_key,
        content_sha256=content_sha256,
        ingest_state=decision.ingest_state,
        replaced_existing=False,
        collection_name=collection_name,
        embedding_model=embedding_model,
        processing_version=identity.processing_version.serialize(),
        metadata={**dict(metadata or {}), "skip_mode": "fast"},
    )


def build_indexed_ingested_document(
    *,
    prepared: PreparedDocument,
    identity: ResolvedIngestIdentity,
    decision: IngestDecision,
    result: IndexResult,
    collection: str,
    namespace: str,
    collection_name: str,
    embedding_model: str,
    metadata: dict[str, str] | None,
) -> IngestedDocument:
    return build_ingested_document(
        prepared=prepared,
        document_id=identity.document_id,
        collection=collection,
        namespace=namespace,
        chunk_count=result.chunk_count,
        document_key=result.document_key,
        content_sha256=result.content_sha256,
        ingest_state=decision.ingest_state,
        replaced_existing=decision.existing is not None,
        collection_name=collection_name,
        embedding_model=embedding_model,
        processing_version=identity.processing_version,
        metadata=metadata,
    )


def _prepared_with_existing_filename(
    prepared: PreparedDocument,
    document_key: str | None,
) -> PreparedDocument:
    if document_key is None or not document_key.strip():
        return prepared
    filename = filename_from_document_key(document_key, fallback=prepared.filename)
    if not filename:
        return prepared
    return replace(prepared, filename=filename)
