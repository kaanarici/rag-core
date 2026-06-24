"""In-memory document-record lookup helpers."""

from __future__ import annotations

from collections.abc import Iterable

from rag_core.search.policy import VectorStorePolicy
from rag_core.search.document_records import (
    payload_match_str,
    resolve_document_id_from_payload,
    stored_document_record_from_payload,
    validate_document_lookup_inputs,
)
from rag_core.search.request_models import StoredDocumentRecord

from .memory_query_scoring import MemoryPoint


def get_memory_document_record(
    points: Iterable[MemoryPoint],
    *,
    namespace: str,
    collection: str,
    document_id: str | None,
    document_key: str | None,
    policy: VectorStorePolicy,
) -> StoredDocumentRecord | None:
    namespace_scoped, collection_scoped = validate_document_lookup_inputs(
        namespace=namespace,
        collection=collection,
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
            collection=collection_scoped,
            document_id=document_id,
            document_key=document_key,
            policy=policy,
        )
    ]
    if not matches:
        return None

    sample = matches[0].payload
    resolved_document_id = resolve_document_id_from_payload(
        payload=sample,
        document_id_field=policy.document_id_field,
        fallback_document_id=document_id,
        invalid_message=(
            f"memory document record payload field {policy.document_id_field!r} "
            "must be a string"
        ),
    )
    if not resolved_document_id:
        return None
    chunk_count = sum(
        1
        for stored in all_points
        if _matches_document_lookup(
            stored,
            namespace=namespace_scoped,
            collection=collection_scoped,
            document_id=resolved_document_id,
            document_key=None,
            policy=policy,
        )
    )
    return stored_document_record_from_payload(
        payload=sample,
        namespace=namespace_scoped,
        collection=collection_scoped,
        document_id=resolved_document_id,
        chunk_count=chunk_count,
        policy=policy,
        invalid_field_message=(
            "memory document record payload field {field!r} must be a string"
        ),
    )


def _matches_document_lookup(
    stored: MemoryPoint,
    *,
    namespace: str,
    collection: str,
    document_id: str | None,
    document_key: str | None,
    policy: VectorStorePolicy,
) -> bool:
    payload = stored.payload
    if payload_match_str(payload, policy.namespace_field) != namespace:
        return False
    if payload_match_str(payload, policy.collection_field) != collection:
        return False
    if document_id is not None:
        if payload_match_str(payload, policy.document_id_field) != document_id:
            return False
    if document_key is not None:
        if payload_match_str(payload, policy.document_key_field) != document_key:
            return False
    return True
