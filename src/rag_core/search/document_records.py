"""Shared document-record lookup validation and payload extraction."""

from __future__ import annotations

from collections.abc import Mapping

from rag_core.search.policy import VectorStorePolicy
from rag_core.search.request_models import StoredDocumentRecord


def validate_document_lookup_inputs(
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


def resolve_document_id_from_payload(
    *,
    payload: Mapping[str, object],
    document_id_field: str,
    fallback_document_id: str | None,
    invalid_message: str,
    reject_blank: bool = False,
) -> str | None:
    value = payload.get(document_id_field)
    if value is None:
        return fallback_document_id
    if not isinstance(value, str) or (reject_blank and not value.strip()):
        raise ValueError(invalid_message)
    return value


def stored_document_record_from_payload(
    *,
    payload: Mapping[str, object],
    namespace: str,
    corpus_id: str,
    document_id: str,
    chunk_count: int,
    policy: VectorStorePolicy,
    invalid_field_message: str,
) -> StoredDocumentRecord:
    return StoredDocumentRecord(
        document_id=document_id,
        namespace=namespace,
        corpus_id=corpus_id,
        document_key=payload_optional_str(
            payload,
            policy.document_key_field,
            invalid_field_message=invalid_field_message,
        ),
        content_sha256=payload_optional_str(
            payload,
            policy.content_sha256_field,
            invalid_field_message=invalid_field_message,
        ),
        processing_version=payload_optional_str(
            payload,
            policy.processing_version_field,
            invalid_field_message=invalid_field_message,
        ),
        chunk_count=chunk_count,
    )


def payload_optional_str(
    payload: Mapping[str, object],
    key: str,
    *,
    invalid_field_message: str,
) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(invalid_field_message.format(field=key))
    return value


def payload_match_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        return ""
    return value


__all__ = [
    "payload_match_str",
    "payload_optional_str",
    "resolve_document_id_from_payload",
    "stored_document_record_from_payload",
    "validate_document_lookup_inputs",
]
