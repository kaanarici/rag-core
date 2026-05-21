from __future__ import annotations

import hashlib

from rag_core.core_models import ProcessingFingerprint
from rag_core.search.policy import DEFAULT_POLICY, VectorStorePolicy
from rag_core.search.types import StoredDocumentRecord


def compute_content_sha256(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


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
    corpus_id: str,
    document_key: str,
    document_id: str | None,
    policy: VectorStorePolicy = DEFAULT_POLICY,
) -> str:
    if document_id and document_id.strip():
        return document_id.strip()
    return policy.make_document_id(
        namespace=namespace,
        corpus_id=corpus_id,
        document_key=document_key,
    )


def resolve_ingest_state(
    existing: StoredDocumentRecord | None,
    *,
    content_sha256: str,
    processing_version: ProcessingFingerprint,
    force_reindex: bool = False,
) -> tuple[str, bool]:
    if existing is None:
        return "created", True
    if existing.content_sha256 == content_sha256:
        if force_reindex:
            return "reindexed", True
        if existing.processing_version != processing_version.serialize():
            return "reindexed", True
        return "unchanged", False
    return "replaced", True
