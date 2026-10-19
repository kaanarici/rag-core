from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json

from rag_core.config.ingest_config import DEFAULT_INGEST_SOURCE_TYPE
from rag_core.core_models import ProcessingFingerprint
from rag_core.file_io import compute_content_sha256
from rag_core.search.policy import DEFAULT_POLICY, VectorStorePolicy
from rag_core.search.request_models import StoredDocumentRecord
from rag_core.documents.converters import converter_settings
from rag_core.documents.formats import registered_converter_key
from rag_core.ingest.states import (
    INGEST_STATE_CREATED,
    INGEST_STATE_REINDEXED,
    INGEST_STATE_REPLACED,
    INGEST_STATE_UNCHANGED,
    IngestState,
)


@dataclass(frozen=True)
class ResolvedIngestIdentity:
    source_type: str
    processing_version: ProcessingFingerprint
    document_key: str
    document_id: str
    content_sha256: str
    ingest_fingerprint: str


def resolve_ingest_identity(
    *,
    default_source_type: str,
    source_type: str | None,
    processing_version: ProcessingFingerprint,
    file_bytes: bytes,
    filename: str,
    mime_type: str,
    path: str | None,
    document_key: str | None,
    document_id: str | None,
    metadata: dict[str, str] | None,
    tenant_id: str,
    collection: str,
    policy: VectorStorePolicy,
    embedding_model: str | None = None,
    embedding_dimensions: int | None = None,
    ocr_identity: tuple[str, str | None] | None = None,
) -> ResolvedIngestIdentity:
    resolved_source_type = (
        source_type if source_type is not None else default_source_type
    ).strip() or DEFAULT_INGEST_SOURCE_TYPE
    resolved_processing_version = replace(
        processing_version, source_type=resolved_source_type
    )
    resolved_document_key = resolve_document_key(
        filename=filename,
        path=path,
        document_key=document_key,
    )
    resolved_document_id = resolve_document_id(
        tenant_id=tenant_id,
        collection=collection,
        document_key=resolved_document_key,
        document_id=document_id,
        policy=policy,
    )
    content_sha256 = compute_content_sha256(file_bytes)
    return ResolvedIngestIdentity(
        source_type=resolved_source_type,
        processing_version=resolved_processing_version,
        document_key=resolved_document_key,
        document_id=resolved_document_id,
        content_sha256=content_sha256,
        ingest_fingerprint=_ingest_fingerprint(
            content_sha256=content_sha256,
            processing_version=resolved_processing_version,
            document_key=resolved_document_key,
            filename=filename,
            mime_type=mime_type,
            path=path,
            source_type=resolved_source_type,
            metadata=metadata,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            ocr_identity=ocr_identity,
        ),
    )


def _ingest_fingerprint(
    *,
    content_sha256: str,
    processing_version: ProcessingFingerprint,
    document_key: str,
    filename: str,
    mime_type: str,
    path: str | None,
    source_type: str,
    metadata: dict[str, str] | None,
    embedding_model: str | None,
    embedding_dimensions: int | None,
    ocr_identity: tuple[str, str | None] | None,
) -> str:
    payload = {
        "content_sha256": content_sha256,
        "processing_version": processing_version.serialize(),
        "document_key": document_key,
        "filename": filename,
        "mime_type": mime_type,
        "path": path,
        "source_type": source_type,
        "metadata": dict(sorted((metadata or {}).items())),
        "embedding": {"model": embedding_model, "dimensions": embedding_dimensions},
        "ocr": ocr_identity,
        "parser_settings": converter_settings(
            registered_converter_key(filename=filename, mime_type=mime_type) or "text"
        ),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_document_key(
    *,
    filename: str,
    path: str | None,
    document_key: str | None,
) -> str:
    if document_key and document_key.strip():
        return document_key.strip()
    if path and path.strip():
        return path.strip()
    return filename.strip()


def resolve_document_id(
    *,
    tenant_id: str,
    collection: str,
    document_key: str,
    document_id: str | None,
    policy: VectorStorePolicy = DEFAULT_POLICY,
) -> str:
    if document_id and document_id.strip():
        return document_id.strip()
    return policy.make_document_id(
        tenant_id=tenant_id,
        collection=collection,
        document_key=document_key,
    )


def resolve_ingest_state(
    existing: StoredDocumentRecord | None,
    *,
    content_sha256: str,
    ingest_fingerprint: str | None = None,
    processing_version: ProcessingFingerprint,
    force_reindex: bool = False,
) -> tuple[IngestState, bool]:
    if existing is None:
        return INGEST_STATE_CREATED, True
    if not existing.is_complete:
        return INGEST_STATE_REINDEXED, True
    if existing.content_sha256 == content_sha256:
        if force_reindex:
            return INGEST_STATE_REINDEXED, True
        if existing.processing_version != processing_version.serialize():
            return INGEST_STATE_REINDEXED, True
        if (
            existing.ingest_fingerprint is not None
            and ingest_fingerprint is not None
            and existing.ingest_fingerprint != ingest_fingerprint
        ):
            return INGEST_STATE_REINDEXED, True
        return INGEST_STATE_UNCHANGED, False
    return INGEST_STATE_REPLACED, True
