"""In-memory document-record lookup helpers."""

from __future__ import annotations

from collections.abc import Iterable

from rag_core.search.policy import VectorStorePolicy
from rag_core.search.request_models import StoredDocumentRecord

from .memory_query_scoring import MemoryPoint


def get_memory_document_record(
    points: Iterable[MemoryPoint],
    *,
    namespace: str,
    corpus_id: str,
    document_id: str | None,
    document_key: str | None,
    policy: VectorStorePolicy,
) -> StoredDocumentRecord | None:
    namespace_scoped, corpus_scoped = _validate_document_lookup_inputs(
        namespace=namespace,
        corpus_id=corpus_id,
        document_id=document_id,
        document_key=document_key,
    )
    all_points = tuple(points)
    matches = [
        stored
        for stored in all_points
        if _matches_document_lookup(
            stored,
            namespace=namespace_scoped,
            corpus_id=corpus_scoped,
            document_id=document_id,
            document_key=document_key,
            policy=policy,
        )
    ]
    if not matches:
        return None

    sample = matches[0].payload
    resolved_document_id = _payload_required_str(
        sample,
        policy.document_id_field,
        fallback=document_id,
    )
    if not resolved_document_id:
        return None
    chunk_count = sum(
        1
        for stored in all_points
        if _matches_document_lookup(
            stored,
            namespace=namespace_scoped,
            corpus_id=corpus_scoped,
            document_id=resolved_document_id,
            document_key=None,
            policy=policy,
        )
    )
    return StoredDocumentRecord(
        document_id=resolved_document_id,
        namespace=namespace_scoped,
        corpus_id=corpus_scoped,
        document_key=_payload_str(sample, policy.document_key_field),
        content_sha256=_payload_str(sample, policy.content_sha256_field),
        processing_version=_payload_str(sample, policy.processing_version_field),
        chunk_count=chunk_count,
    )


def _validate_document_lookup_inputs(
    *,
    namespace: str,
    corpus_id: str,
    document_id: str | None,
    document_key: str | None,
) -> tuple[str, str]:
    namespace_scoped = namespace.strip()
    if not namespace_scoped:
        raise ValueError("namespace is required for get_document_record")
    corpus_scoped = corpus_id.strip()
    if not corpus_scoped:
        raise ValueError("corpus_id is required for get_document_record")
    if document_id is None and document_key is None:
        raise ValueError(
            "document_id or document_key is required for get_document_record"
        )
    return namespace_scoped, corpus_scoped


def _matches_document_lookup(
    stored: MemoryPoint,
    *,
    namespace: str,
    corpus_id: str,
    document_id: str | None,
    document_key: str | None,
    policy: VectorStorePolicy,
) -> bool:
    payload = stored.payload
    if _payload_match_str(payload, policy.namespace_field) != namespace:
        return False
    if _payload_match_str(payload, policy.corpus_id_field) != corpus_id:
        return False
    if document_id is not None:
        if _payload_match_str(payload, policy.document_id_field) != document_id:
            return False
    if document_key is not None:
        if _payload_match_str(payload, policy.document_key_field) != document_key:
            return False
    return True


def _payload_str(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"memory document record payload field {key!r} must be a string"
        )
    return value


def _payload_match_str(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        return ""
    return value


def _payload_required_str(
    payload: dict[str, object],
    key: str,
    *,
    fallback: str | None,
) -> str | None:
    value = payload.get(key, _MISSING)
    if value is _MISSING or value is None:
        return fallback
    if not isinstance(value, str):
        raise ValueError(
            f"memory document record payload field {key!r} must be a string"
        )
    return value


_MISSING = object()
