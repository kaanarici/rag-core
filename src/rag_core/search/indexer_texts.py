from __future__ import annotations

from dataclasses import dataclass

from rag_core.config import INGEST_SOURCE_TYPE_URL, PRECHUNKED_CHUNKING_STRATEGY
from rag_core.core_models import PreparedChunk
from rag_core.documents.prepare_chunks import (
    prepare_pre_chunked_texts,
    prepare_text_chunks,
)
from rag_core.documents.chunking.router import is_code_content
from rag_core.search.text_builder import build_sparse_text, build_textual_representation
from rag_core.search.vector_models import ContentType

from .indexer_models import IndexRequest


@dataclass(frozen=True)
class IndexTextSet:
    content_type: ContentType
    chunks: list[PreparedChunk]
    dense_texts: list[str]
    payload_texts: list[str]
    sparse_texts: list[str]


def build_index_texts(req: IndexRequest) -> IndexTextSet:
    content_type = resolve_content_type(req.mime_type, req.filename)
    chunks = resolve_chunks(req)
    if not chunks:
        return IndexTextSet(
            content_type=content_type,
            chunks=[],
            dense_texts=[],
            payload_texts=[],
            sparse_texts=[],
        )
    return IndexTextSet(
        content_type=content_type,
        chunks=chunks,
        dense_texts=_build_dense_texts(
            req=req,
            content_type=content_type,
            chunks=chunks,
        ),
        payload_texts=_build_payload_texts(
            req=req,
            content_type=content_type,
            chunks=chunks,
        ),
        sparse_texts=_build_sparse_texts(req=req, chunks=chunks),
    )


def resolve_content_type(mime_type: str, filename: str) -> ContentType:
    return (
        ContentType.CODE
        if is_code_content(
            mime_type=mime_type, filename=filename, allow_text_x_prefix=True
        )
        else ContentType.DOCUMENT
    )


def resolve_chunks(req: IndexRequest) -> list[PreparedChunk]:
    if req.prepared_chunks:
        return list(req.prepared_chunks)
    if req.pre_chunked_texts:
        return prepare_pre_chunked_texts(
            req.pre_chunked_texts,
            embedding_texts=req.embedding_chunk_texts,
            chunk_metadata=req.chunk_metadata,
            chunking_strategy=req.chunker_strategy or PRECHUNKED_CHUNKING_STRATEGY,
        )
    return prepare_text_chunks(
        req.text,
        mime_type=req.mime_type,
        filename=req.filename,
    )


def _build_dense_texts(
    *,
    req: IndexRequest,
    content_type: ContentType,
    chunks: list[PreparedChunk],
) -> list[str]:
    return [
        build_textual_representation(
            content=chunk.embedding_text,
            source_type=req.source_type,
            name=req.filename,
            content_type=content_type,
            path=_textual_path(req),
            extra_fields=req.extra_fields,
        )
        for chunk in chunks
    ]


def _build_payload_texts(
    *,
    req: IndexRequest,
    content_type: ContentType,
    chunks: list[PreparedChunk],
) -> list[str]:
    del req, content_type
    return [chunk.text for chunk in chunks]


def _build_sparse_texts(
    *,
    req: IndexRequest,
    chunks: list[PreparedChunk],
) -> list[str]:
    metadata: dict[str, str] = {
        "source_type": req.source_type,
        "filename": req.filename,
        "document_id": req.document_id,
    }
    if req.extra_fields:
        metadata.update(req.extra_fields)
    return [
        build_sparse_text(
            chunk_text=chunk.sparse_text or chunk.text,
            metadata=metadata,
        )
        for chunk in chunks
    ]


def _textual_path(req: IndexRequest) -> str | None:
    if req.source_type == INGEST_SOURCE_TYPE_URL:
        return req.document_path or req.path
    return req.document_key
