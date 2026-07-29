from __future__ import annotations

from rag_core.core_models import ProcessingFingerprint
from rag_core.ingest.states import (
    INGEST_STATE_CREATED,
    INGEST_STATE_REINDEXED,
    INGEST_STATE_REPLACED,
    INGEST_STATE_UNCHANGED,
    IngestState,
)
from rag_core.search.policy import DEFAULT_POLICY, VectorStorePolicy
from rag_core.search.request_models import StoredDocumentRecord


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
    namespace: str,
    collection: str,
    document_key: str,
    document_id: str | None,
    policy: VectorStorePolicy = DEFAULT_POLICY,
) -> str:
    if document_id and document_id.strip():
        return document_id.strip()
    return policy.make_document_id(
        namespace=namespace,
        collection=collection,
        document_key=document_key,
    )


def resolve_ingest_state(
    existing: StoredDocumentRecord | None,
    *,
    content_sha256: str,
    processing_version: ProcessingFingerprint,
    force_reindex: bool = False,
) -> tuple[IngestState, bool]:
    if existing is None:
        return INGEST_STATE_CREATED, True
    if existing.content_sha256 == content_sha256:
        if force_reindex:
            return INGEST_STATE_REINDEXED, True
        if existing.processing_version != processing_version.serialize():
            return INGEST_STATE_REINDEXED, True
        return INGEST_STATE_UNCHANGED, False
    return INGEST_STATE_REPLACED, True
