"""Qdrant document-record lookup helpers."""

from __future__ import annotations

from qdrant_client import AsyncQdrantClient

from rag_core.search.policy import VectorStorePolicy
from rag_core.search.types import StoredDocumentRecord

from .qdrant_filters import build_document_count_filter, build_document_lookup_filter
from .qdrant_payloads import (
    _build_stored_document_record,
    _count_result_value,
    _resolve_document_id,
    _validate_document_lookup_inputs,
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
    namespace_scoped, corpus_scoped = _validate_document_lookup_inputs(
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
    resolved_document_id = _resolve_document_id(
        payload=payload,
        fallback_document_id=document_id,
        policy=policy,
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
    return _build_stored_document_record(
        payload=payload,
        namespace=namespace_scoped,
        corpus_id=corpus_scoped,
        document_id=resolved_document_id,
        chunk_count=_count_result_value(getattr(chunk_count, "count", None)),
        policy=policy,
    )
