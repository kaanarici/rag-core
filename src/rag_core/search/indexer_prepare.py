"""Indexing prepare stage: dense/sparse embedding vectors, index texts, and embedding orchestration."""

from __future__ import annotations

from rag_core._async import run_sync
from collections.abc import Sequence

import logging
from rag_core.search.provider_protocols import (
    EmbeddingProvider,
    SparseEmbedder,
    provider_name as provider_name,
)
from rag_core.search.sparse_channels import single_sparse_channel
from rag_core.search.vector_models import SparseVector
from dataclasses import dataclass
from rag_core.config import INGEST_SOURCE_TYPE_URL
from rag_core.core_models import PreparedChunk
from rag_core.documents.prepare_chunks import (
    prepare_text_chunks,
)
from rag_core.documents.chunking.router import is_code_content
from rag_core.search.text_builder import build_sparse_text, build_textual_representation
from rag_core.search.vector_models import ContentType
from .indexer_models import IndexRequest
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, TypeVar
from rag_core.config.embedding_config import DEFAULT_EMBEDDING_BATCH_SIZE
from rag_core.events.emit import emit_event, now_ms
from rag_core.events.types import EmbedCompleted, EmbedRequested
from rag_core.retrieval_channels import (
    RetrievalChannel,
    DENSE_RETRIEVAL_CHANNEL,
    SPARSE_RETRIEVAL_CHANNEL,
)


T = TypeVar("T")

logger = logging.getLogger(__name__)


async def embed_dense_texts(
    embedding_provider: EmbeddingProvider, texts: list[str], *, batch_size: int
) -> list[list[float]]:
    validate_embedding_batch_size(batch_size)
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        result = await embedding_provider.embed_texts(batch)
        validate_dense_vectors(
            vectors=result,
            expected_count=len(batch),
            expected_dimensions=embedding_provider.dimensions,
            provider_name=provider_name(embedding_provider),
            model_name=embedding_provider.model_name,
        )
        vectors.extend(result)
    return vectors


def validate_embedding_batch_size(batch_size: int) -> int:
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("embedding_batch_size must be a positive integer")
    return batch_size


def validate_dense_vectors(
    *,
    vectors: list[list[float]],
    expected_count: int,
    expected_dimensions: int,
    provider_name: str,
    model_name: str,
) -> None:
    if expected_dimensions <= 0:
        raise ValueError("embedding provider dimensions must be positive")
    if len(vectors) != expected_count:
        raise ValueError(
            "Dense embedding count mismatch for %s/%s: expected %d vectors, got %d"
            % (provider_name, model_name, expected_count, len(vectors))
        )
    for index, vector in enumerate(vectors):
        if len(vector) != expected_dimensions:
            raise ValueError(
                "Dense embedding dimension mismatch at chunk index %d for %s/%s: "
                "expected %d dimensions, got %d"
                % (
                    index,
                    provider_name,
                    model_name,
                    expected_dimensions,
                    len(vector),
                )
            )


def embed_sparse_channels(
    *, sparse_embedder: SparseEmbedder, texts: list[str], expected_count: int
) -> list[dict[str, SparseVector]]:
    vectors = sparse_embedder.embed_texts(texts)
    if len(vectors) != expected_count:
        raise ValueError(
            f"Sparse embedding count mismatch: expected {expected_count} got {len(vectors)}"
        )
    return [single_sparse_channel(vector) for vector in vectors]


def resolve_content_type(mime_type: str, filename: str) -> ContentType:
    return (
        ContentType.CODE
        if is_code_content(
            mime_type=mime_type, filename=filename, allow_text_x_prefix=True
        )
        else ContentType.DOCUMENT
    )


def resolve_chunks(req: IndexRequest) -> Sequence[PreparedChunk]:
    if req.prepared_chunks is not None:
        return req.prepared_chunks
    return prepare_text_chunks(req.text, mime_type=req.mime_type, filename=req.filename)


def _build_dense_texts(
    *,
    req: IndexRequest,
    content_type: ContentType,
    chunks: Sequence[PreparedChunk],
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


def _build_sparse_texts(
    *,
    req: IndexRequest,
    chunks: Sequence[PreparedChunk],
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


if TYPE_CHECKING:
    from rag_core.events.sink import EventSink


@dataclass(frozen=True)
class PreparedIndexData:
    content_type: ContentType
    chunks: Sequence[PreparedChunk]
    dense_vectors: list[list[float]]
    sparse_texts: list[str]
    sparse_channels: list[dict[str, SparseVector]]


async def prepare_index_data(
    *,
    req: IndexRequest,
    embedding_provider: EmbeddingProvider,
    sparse_embedder: SparseEmbedder | None,
    event_sink: "EventSink | None" = None,
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    chunks: Sequence[PreparedChunk] | None = None,
) -> PreparedIndexData:
    validate_embedding_batch_size(embedding_batch_size)
    chunks = resolve_chunks(req) if chunks is None else chunks
    content_type = resolve_content_type(req.mime_type, req.filename)
    if not chunks:
        return PreparedIndexData(
            content_type=content_type,
            chunks=chunks,
            dense_vectors=[],
            sparse_texts=[],
            sparse_channels=[],
        )
    dense_texts = _build_dense_texts(req=req, content_type=content_type, chunks=chunks)
    sparse_texts = _build_sparse_texts(req=req, chunks=chunks)

    async def dense() -> list[list[float]]:
        return await embed_dense_texts(
            embedding_provider, dense_texts, batch_size=embedding_batch_size
        )

    dense_vectors = await _embed_with_events(
        dense,
        provider=embedding_provider,
        role=DENSE_RETRIEVAL_CHANNEL,
        count=len(chunks),
        event_sink=event_sink,
    )
    sparse_channels: list[dict[str, SparseVector]]
    if sparse_embedder is None:
        sparse_channels = [{} for _ in chunks]
    else:

        async def sparse() -> list[dict[str, SparseVector]]:
            return await run_sync(
                embed_sparse_channels,
                sparse_embedder=sparse_embedder,
                texts=sparse_texts,
                expected_count=len(chunks),
            )

        sparse_channels = await _embed_with_events(
            sparse,
            provider=sparse_embedder,
            role=SPARSE_RETRIEVAL_CHANNEL,
            count=len(chunks),
            event_sink=event_sink,
        )
    return PreparedIndexData(
        content_type=content_type,
        chunks=chunks,
        dense_vectors=dense_vectors,
        sparse_texts=sparse_texts,
        sparse_channels=sparse_channels,
    )


async def _embed_with_events(
    run: Callable[[], Awaitable[T]],
    *,
    provider: object,
    role: RetrievalChannel,
    count: int,
    event_sink: EventSink | None,
) -> T:
    name, model = provider_name(provider), getattr(provider, "model_name", "")
    emit_event(
        event_sink,
        EmbedRequested(provider=name, model=model, text_count=count, role=role),
    )
    started = now_ms()
    result = await run()
    emit_event(
        event_sink,
        EmbedCompleted(
            provider=name,
            model=model,
            text_count=count,
            role=role,
            duration_ms=now_ms() - started,
        ),
    )
    return result
