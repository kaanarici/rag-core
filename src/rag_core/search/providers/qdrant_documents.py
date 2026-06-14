"""Qdrant document-record lookup helpers."""

from __future__ import annotations

from collections.abc import Sequence

from qdrant_client import AsyncQdrantClient

from rag_core.search.document_records import (
    resolve_document_id_from_payload,
    stored_document_record_from_payload,
    validate_document_lookup_inputs,
)
from rag_core.search.policy import VectorStorePolicy
from rag_core.search.request_models import StoredDocumentRecord
from rag_core.search.vector_models import SearchResult

from .chunk_lookup import validate_chunk_lookup_inputs
from .qdrant_filters import (
    build_chunk_index_lookup_filter,
    build_document_count_filter,
    build_document_lookup_filter,
)
from .qdrant_payloads import (
    _count_result_value,
    _record_to_result,
)


async def get_qdrant_document_record(
    *,
    client: AsyncQdrantClient,
    collection_name: str,
    namespace: str,
    corpus_id: str,
    document_id: str | None,
    document_key: str | None,
    policy: VectorStorePolicy,
) -> StoredDocumentRecord | None:
    namespace_scoped, corpus_scoped = validate_document_lookup_inputs(
        namespace=namespace,
        corpus_id=corpus_id,
        document_id=document_id,
        document_key=document_key,
    )

    records, _ = await client.scroll(
        collection_name=collection_name,
        scroll_filter=build_document_lookup_filter(
            namespace=namespace_scoped,
            corpus_id=corpus_scoped,
            document_id=document_id,
            document_key=document_key,
            policy=policy,
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not records:
        return None

    payload = records[0].payload or {}
    resolved_document_id = resolve_document_id_from_payload(
        payload=payload,
        document_id_field=policy.document_id_field,
        fallback_document_id=document_id,
        invalid_message=(
            f"qdrant document record payload field {policy.document_id_field!r} "
            "must be a string"
        ),
    )
    if not resolved_document_id:
        return None
    chunk_count = await client.count(
        collection_name=collection_name,
        count_filter=build_document_count_filter(
            namespace=namespace_scoped,
            corpus_id=corpus_scoped,
            document_id=resolved_document_id,
            policy=policy,
        ),
        exact=True,
    )
    return stored_document_record_from_payload(
        payload=payload,
        namespace=namespace_scoped,
        corpus_id=corpus_scoped,
        document_id=resolved_document_id,
        chunk_count=_count_result_value(getattr(chunk_count, "count", None)),
        policy=policy,
        invalid_field_message=(
            "qdrant document record payload field {field!r} must be a string"
        ),
    )


async def get_qdrant_chunks_by_index(
    *,
    client: AsyncQdrantClient,
    collection_name: str,
    namespace: str,
    corpus_id: str,
    document_id: str,
    chunk_indices: Sequence[int],
    policy: VectorStorePolicy,
) -> list[SearchResult]:
    namespace_scoped, corpus_scoped, document_scoped, indices = (
        validate_chunk_lookup_inputs(
            namespace=namespace,
            corpus_id=corpus_id,
            document_id=document_id,
            chunk_indices=chunk_indices,
        )
    )
    if not indices:
        return []

    records, _ = await client.scroll(
        collection_name=collection_name,
        scroll_filter=build_chunk_index_lookup_filter(
            namespace=namespace_scoped,
            corpus_id=corpus_scoped,
            document_id=document_scoped,
            chunk_indices=indices,
            policy=policy,
        ),
        limit=len(indices),
        with_payload=True,
        with_vectors=False,
    )
    results = [_record_to_result(record, policy=policy) for record in records]
    return sorted(results, key=lambda result: result.chunk_index or 0)
