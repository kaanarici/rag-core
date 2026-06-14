from __future__ import annotations

from rag_core.config import (
    CONTENT_CHUNKER_CHUNKING_STRATEGY,
    INGEST_SOURCE_TYPE_URL,
    PRECHUNKED_CHUNKING_STRATEGY,
)
from rag_core.search.policy import DEFAULT_POLICY, VectorStorePolicy
from rag_core.search.sparse_channels import primary_sparse_channel
from rag_core.search.stored_payload import build_stored_payload
from rag_core.search.vector_models import ContentType, VectorPoint

from .indexer_embeddings import PreparedIndexData
from .indexer_models import IndexRequest
from .indexer_sections import build_section_lookup, resolve_section_info


def build_points(
    *,
    req: IndexRequest,
    namespace: str,
    prepared: PreparedIndexData,
    policy: VectorStorePolicy = DEFAULT_POLICY,
) -> tuple[list[VectorPoint], list[str]]:
    section_lookup = build_section_lookup(req.section_mappings)
    points: list[VectorPoint] = []
    point_ids: list[str] = []

    for index, chunk in enumerate(prepared.chunks):
        point_id = make_point_id(
            namespace=namespace,
            corpus_id=req.corpus_id,
            document_id=req.document_id,
            chunk_index=index,
            policy=policy,
        )
        sparse_channels = prepared.sparse_channels[index]
        primary_sparse = primary_sparse_channel(
            sparse_channels,
            missing_message="Sparse embedding is missing for chunk index %d" % index,
        )
        payload = _build_payload(
            req=req,
            namespace=namespace,
            chunk_index=index,
            chunk_text=chunk.text,
            chunk_token_count=chunk.token_count,
            chunk_start_char=chunk.start_char,
            chunk_end_char=chunk.end_char,
            payload_text=prepared.payload_texts[index],
            content_type=prepared.content_type,
            filter_metadata=_filter_metadata(
                document_metadata=req.document_metadata,
                extra_fields=req.extra_fields,
                chunk_metadata=chunk.metadata,
            ),
            section_info=resolve_section_info(
                chunk_metadata=chunk.metadata,
                mapping=section_lookup.get(index),
            ),
            policy=policy,
        )
        points.append(
            VectorPoint(
                id=point_id,
                dense_vector=prepared.dense_vectors[index],
                sparse_vector=primary_sparse,
                sparse_vectors=dict(sparse_channels),
                payload=payload,
                sparse_text=prepared.sparse_texts[index],
            )
        )
        point_ids.append(point_id)

    return points, point_ids


def _build_payload(
    *,
    req: IndexRequest,
    namespace: str,
    chunk_index: int,
    chunk_text: str,
    chunk_token_count: int,
    chunk_start_char: int | None,
    chunk_end_char: int | None,
    payload_text: str,
    content_type: ContentType,
    filter_metadata: dict[str, object],
    section_info: dict[str, object] | None,
    policy: VectorStorePolicy,
) -> dict[str, object]:
    document_path = _stored_document_path(req)
    chunker_strategy = req.chunker_strategy or (
        PRECHUNKED_CHUNKING_STRATEGY
        if req.pre_chunked_texts
        else CONTENT_CHUNKER_CHUNKING_STRATEGY
    )
    return build_stored_payload(
        namespace=namespace,
        corpus_id=req.corpus_id,
        document_id=req.document_id,
        document_key=req.document_key,
        content_sha256=req.content_sha256,
        processing_version=req.processing_version,
        filename=req.filename,
        mime_type=req.mime_type,
        source_type=req.source_type,
        document_path=document_path,
        chunk_index=chunk_index,
        chunk_text=chunk_text,
        chunk_token_count=chunk_token_count,
        chunk_start_char=chunk_start_char,
        chunk_end_char=chunk_end_char,
        payload_text=payload_text,
        content_type=content_type,
        embedding_model=req.embedding_model,
        chunker_strategy=chunker_strategy,
        title=_resolve_display_title(req),
        filter_metadata=filter_metadata,
        section_info=section_info,
        policy=policy,
    )


def _stored_document_path(req: IndexRequest) -> str | None:
    if req.source_type == INGEST_SOURCE_TYPE_URL:
        return req.document_path or req.path
    return None


def _filter_metadata(
    *,
    document_metadata: dict[str, object] | None,
    extra_fields: dict[str, str] | None,
    chunk_metadata: object,
) -> dict[str, object]:
    metadata: dict[str, object] = {}
    if document_metadata:
        metadata.update(document_metadata)
    if extra_fields:
        metadata.update(extra_fields)
    if isinstance(chunk_metadata, dict):
        metadata.update(chunk_metadata)
    return metadata


def _resolve_display_title(req: IndexRequest) -> str:
    raw_title = (req.extra_fields or {}).get("title")
    if raw_title is None:
        return req.filename
    title = str(raw_title).strip()
    return title or req.filename


def make_point_id(
    *,
    namespace: str,
    corpus_id: str,
    document_id: str,
    chunk_index: int,
    policy: VectorStorePolicy = DEFAULT_POLICY,
) -> str:
    return policy.make_point_id(
        namespace=namespace,
        corpus_id=corpus_id,
        document_id=document_id,
        chunk_index=chunk_index,
    )


def build_stale_point_ids(
    req: IndexRequest,
    *,
    new_chunk_count: int,
    policy: VectorStorePolicy,
) -> list[str]:
    if not req.document_id or req.existing_chunk_count is None:
        return []
    if req.existing_chunk_count <= new_chunk_count:
        return []
    return [
        make_point_id(
            namespace=req.namespace,
            corpus_id=req.corpus_id,
            document_id=req.document_id,
            chunk_index=index,
            policy=policy,
        )
        for index in range(new_chunk_count, req.existing_chunk_count)
    ]
