"""Build vector-store points (and section locators) for an indexed document."""

from __future__ import annotations

from collections.abc import Mapping
from rag_core.search.stored_payload import SECTION_PAYLOAD_KEYS
from rag_core.config import (
    INGEST_SOURCE_TYPE_URL,
)
from rag_core.search.policy import DEFAULT_POLICY, VectorStorePolicy
from rag_core.search.stored_payload import build_stored_payload
from rag_core.search.vector_models import ContentType, VectorPoint
from rag_core.core_models import PreparedChunk
from .indexer_prepare import PreparedIndexData
from .indexer_models import IndexRequest


def chunk_locator_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    keys = ("section_id", "section_path", "section_title", *SECTION_PAYLOAD_KEYS)
    return {key: metadata[key] for key in keys if key in metadata}


def with_section_title(mapping: Mapping[str, object]) -> dict[str, object]:
    section_path = mapping.get("section_path")
    section_title = mapping.get("section_title")
    if section_title is None and isinstance(section_path, str) and section_path.strip():
        section_title = section_path.split(">")[-1].strip()
    return {
        **dict(mapping),
        "section_title": section_title,
    }


def build_points(
    *,
    req: IndexRequest,
    tenant_id: str,
    prepared: PreparedIndexData,
    policy: VectorStorePolicy = DEFAULT_POLICY,
) -> list[VectorPoint]:
    points: list[VectorPoint] = []

    for index, chunk in enumerate(prepared.chunks):
        point_id = make_point_id(
            tenant_id=tenant_id,
            collection=req.collection,
            document_id=req.document_id,
            chunk_index=index,
            policy=policy,
        )
        sparse_channels = prepared.sparse_channels[index]
        payload = _build_payload(
            req=req,
            tenant_id=tenant_id,
            chunk_index=index,
            chunk=chunk,
            content_type=prepared.content_type,
            filter_metadata=_filter_metadata(
                document_metadata=req.document_metadata,
                extra_fields=req.extra_fields,
                chunk_metadata=chunk.metadata,
            ),
            section_info=with_section_title(chunk_locator_metadata(chunk.metadata)),
            policy=policy,
        )
        points.append(
            VectorPoint(
                id=point_id,
                dense_vector=prepared.dense_vectors[index],
                payload=payload,
                sparse_vectors=dict(sparse_channels),
                sparse_text=prepared.sparse_texts[index],
            )
        )

    return points


def _build_payload(
    *,
    req: IndexRequest,
    tenant_id: str,
    chunk_index: int,
    chunk: PreparedChunk,
    content_type: ContentType,
    filter_metadata: dict[str, object],
    section_info: dict[str, object] | None,
    policy: VectorStorePolicy,
) -> dict[str, object]:
    document_path = _stored_document_path(req)
    return build_stored_payload(
        tenant_id=tenant_id,
        collection=req.collection,
        document_id=req.document_id,
        document_key=req.document_key,
        content_sha256=req.content_sha256,
        ingest_fingerprint=req.ingest_fingerprint,
        processing_version=req.processing_version,
        filename=req.filename,
        mime_type=req.mime_type,
        source_type=req.source_type,
        document_path=document_path,
        chunk_index=chunk_index,
        chunk_start_char=chunk.start_char,
        chunk_end_char=chunk.end_char,
        payload_text=chunk.text,
        content_type=content_type,
        chunker_strategy=chunk.chunking_strategy,
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
    if isinstance(chunk_metadata, Mapping):
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
    tenant_id: str,
    collection: str,
    document_id: str,
    chunk_index: int,
    policy: VectorStorePolicy = DEFAULT_POLICY,
) -> str:
    return policy.make_point_id(
        tenant_id=tenant_id,
        collection=collection,
        document_id=document_id,
        chunk_index=chunk_index,
    )
