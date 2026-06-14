from __future__ import annotations

from rag_core.core_models import (
    CorpusManifest,
    CorpusManifestEntry,
    IngestedDocument,
    PreparedDocument,
)
from rag_core._engine.core_ocr_metadata import read_ocr_metadata
from rag_core.manifest_entries import sanitize_manifest_metadata


def build_manifest_entry(document: IngestedDocument) -> CorpusManifestEntry:
    ocr = read_ocr_metadata(document.metadata)
    return CorpusManifestEntry(
        document_id=document.document_id,
        namespace=document.namespace,
        corpus_id=document.corpus_id,
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


def build_staged_manifest_entry(
    *,
    prepared: PreparedDocument,
    document_id: str,
    namespace: str,
    corpus_id: str,
    document_key: str | None,
    content_sha256: str,
    filename: str,
    mime_type: str,
    metadata: dict[str, str] | None,
) -> CorpusManifestEntry:
    return CorpusManifestEntry(
        document_id=document_id,
        namespace=namespace,
        corpus_id=corpus_id,
        document_key=document_key,
        content_sha256=content_sha256,
        filename=filename,
        mime_type=mime_type,
        chunk_count=len(prepared.chunks),
        parser=_optional_str(prepared.metadata.get("parser")),
        needs_ocr=bool(prepared.metadata.get("needs_ocr", False)),
        metadata=dict(metadata or {}),
    )


def build_corpus_manifest(
    *,
    namespace: str,
    corpus_id: str,
    collection_name: str,
    embedding_provider: str,
    embedding_model: str,
    embedding_dimensions: int,
    documents: list[IngestedDocument],
) -> CorpusManifest:
    ocr_summaries = [read_ocr_metadata(document.metadata) for document in documents]
    ocr_document_count = sum(1 for ocr in ocr_summaries if ocr.provider is not None)
    ocr_page_count = sum(ocr.page_count for ocr in ocr_summaries)
    entries = tuple(build_manifest_entry(document) for document in documents)
    return CorpusManifest(
        namespace=namespace,
        corpus_id=corpus_id,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
        document_count=len(documents),
        chunk_count=sum(document.chunk_count for document in documents),
        source_document_ids=tuple(document.document_id for document in documents),
        ocr_document_count=ocr_document_count,
        ocr_page_count=ocr_page_count,
        entries=entries,
        documents=tuple(documents),
    )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
