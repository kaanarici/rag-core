from __future__ import annotations

from dataclasses import dataclass, replace

from rag_core.core_lifecycle import (
    compute_content_sha256,
    resolve_document_id,
    resolve_document_key,
)
from rag_core.core_models import ProcessingFingerprint
from rag_core.search.policy import VectorStorePolicy


@dataclass(frozen=True)
class ResolvedIngestIdentity:
    source_type: str
    processing_version: ProcessingFingerprint
    document_key: str
    document_id: str
    content_sha256: str


def resolve_ingest_identity(
    *,
    default_source_type: str,
    source_type: str | None,
    processing_version: ProcessingFingerprint,
    file_bytes: bytes,
    filename: str,
    path: str | None,
    document_key: str | None,
    document_id: str | None,
    namespace: str,
    corpus_id: str,
    policy: VectorStorePolicy,
) -> ResolvedIngestIdentity:
    resolved_source_type = _resolved_source_type(
        default_source_type=default_source_type,
        source_type=source_type,
    )
    resolved_processing_version = _processing_version_for_source(
        processing_version,
        source_type=resolved_source_type,
    )
    resolved_document_key = resolve_document_key(
        filename=filename,
        path=path,
        document_key=document_key,
    )
    resolved_document_id = resolve_document_id(
        namespace=namespace,
        corpus_id=corpus_id,
        document_key=resolved_document_key,
        document_id=document_id,
        policy=policy,
    )
    return ResolvedIngestIdentity(
        source_type=resolved_source_type,
        processing_version=resolved_processing_version,
        document_key=resolved_document_key,
        document_id=resolved_document_id,
        content_sha256=compute_content_sha256(file_bytes),
    )


def _processing_version_for_source(
    processing_version: ProcessingFingerprint,
    *,
    source_type: str,
) -> ProcessingFingerprint:
    if processing_version.source_type == source_type:
        return processing_version
    return replace(processing_version, source_type=source_type)


def _resolved_source_type(
    *,
    default_source_type: str,
    source_type: str | None,
) -> str:
    resolved = source_type if source_type is not None else default_source_type
    return resolved.strip() or "file"
