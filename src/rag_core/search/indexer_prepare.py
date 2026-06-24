"""Indexing prepare stage: dense/sparse embedding vectors, index texts, and embedding orchestration."""

from __future__ import annotations

import logging
from rag_core.search.embedding_cache_diagnostics import EmbeddingCacheCounters
from rag_core.search.embedding_cache_diagnostics import (
    embed_texts_with_cache_observation,
)
from rag_core.search.provider_protocols import (
    EmbeddingProvider,
    SparseEmbedder,
    provider_name as provider_name,
)
from rag_core.search.providers.embedding_cache_models import (
    EMPTY_EMBEDDING_DOCUMENT_SCOPE,
    EmbeddingDocumentScope,
)
from rag_core.search.sparse_channels import single_sparse_channel
from rag_core.search.vector_models import SparseVector
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
from typing import TYPE_CHECKING
from rag_core.config.embedding_config import DEFAULT_EMBEDDING_BATCH_SIZE
from rag_core.events.emit import emit_event, now_ms
from rag_core.events.types import EmbedCompleted, EmbedRequested
from rag_core.retrieval_channels import (
    DENSE_RETRIEVAL_CHANNEL,
    SPARSE_RETRIEVAL_CHANNEL,
)


logger = logging.getLogger(__name__)

async def embed_dense_texts(
    embedding_provider: EmbeddingProvider,
    texts: list[str],
    *,
    processing_fingerprint: str,
    batch_size: int,
    scope: EmbeddingDocumentScope = EMPTY_EMBEDDING_DOCUMENT_SCOPE,
) -> tuple[list[list[float]], EmbeddingCacheCounters]:
    validate_embedding_batch_size(batch_size)
    dense_vectors: list[list[float]] = []
    hits = 0
    misses = 0
    writes = 0
    bypasses = 0
    for index in range(0, len(texts), batch_size):
        batch = texts[index : index + batch_size]
        batch_vectors, counters = await embed_texts_with_cache_observation(
            embedding_provider,
            batch,
            processing_fingerprint=processing_fingerprint,
            scope=scope,
        )
        dense_vectors.extend(batch_vectors)
        hits += counters.hits
        misses += counters.misses
        writes += counters.writes
        bypasses += counters.bypasses
    return dense_vectors, EmbeddingCacheCounters(
        hits=hits,
        misses=misses,
        writes=writes,
        bypasses=bypasses,
    )

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
    *,
    sparse_embedder: SparseEmbedder,
    texts: list[str],
    expected_count: int,
) -> list[dict[str, SparseVector]]:
    sparse_channels = try_embed_sparse_multi(
        sparse_embedder=sparse_embedder, texts=texts
    )
    if sparse_channels is None:
        sparse_vectors = sparse_embedder.embed_texts(texts)
        sparse_channels = [single_sparse_channel(vector) for vector in sparse_vectors]
    if len(sparse_channels) != expected_count:
        raise ValueError(
            "Sparse embedding count mismatch: expected %d got %d"
            % (expected_count, len(sparse_channels))
        )
    return sparse_channels

def try_embed_sparse_multi(
    *,
    sparse_embedder: SparseEmbedder,
    texts: list[str],
) -> list[dict[str, SparseVector]] | None:
    embed_multi = getattr(sparse_embedder, "embed_texts_multi", None)
    if not callable(embed_multi):
        return None
    try:
        raw = embed_multi(texts)
    except Exception as exc:
        logger.warning(
            "Multi-channel sparse embedding failed for %s with %s over %d texts; "
            "using bm25 only",
            provider_name(sparse_embedder),
            type(exc).__name__,
            len(texts),
        )
        return None
    if not isinstance(raw, list) or len(raw) != len(texts):
        return None

    sparse_channels: list[dict[str, SparseVector]] = []
    for item in raw:
        if not isinstance(item, dict):
            return None
        channel_map: dict[str, SparseVector] = {}
        for name, vector in item.items():
            if isinstance(name, str) and name and isinstance(vector, SparseVector):
                channel_map[name] = vector
        if not channel_map:
            return None
        sparse_channels.append(channel_map)
    return sparse_channels


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


if TYPE_CHECKING:
    from rag_core.events.sink import EventSink

@dataclass(frozen=True)
class PreparedIndexData:
    content_type: ContentType
    chunks: list[PreparedChunk]
    dense_vectors: list[list[float]]
    payload_texts: list[str]
    sparse_texts: list[str]
    sparse_channels: list[dict[str, SparseVector]]

async def prepare_index_data(
    *,
    req: IndexRequest,
    embedding_provider: EmbeddingProvider,
    sparse_embedder: SparseEmbedder,
    event_sink: "EventSink | None" = None,
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
) -> PreparedIndexData:
    validate_embedding_batch_size(embedding_batch_size)
    texts = build_index_texts(req)
    if not texts.chunks:
        return PreparedIndexData(
            content_type=texts.content_type,
            chunks=[],
            dense_vectors=[],
            payload_texts=[],
            sparse_texts=[],
            sparse_channels=[],
        )

    dense_provider_name = provider_name(embedding_provider)
    dense_model_name = getattr(embedding_provider, "model_name", "")
    emit_event(
        event_sink,
        EmbedRequested(
            provider=dense_provider_name,
            model=dense_model_name,
            text_count=len(texts.dense_texts),
            role=DENSE_RETRIEVAL_CHANNEL,
        ),
    )
    dense_started_ms = now_ms()
    dense_vectors, dense_cache = await embed_dense_texts(
        embedding_provider,
        texts.dense_texts,
        processing_fingerprint=req.processing_version or "",
        batch_size=embedding_batch_size,
        scope=EmbeddingDocumentScope(
            namespace=req.namespace,
            collection=req.collection,
            document_id=req.document_id,
        ),
    )
    validate_dense_vectors(
        vectors=dense_vectors,
        expected_count=len(texts.chunks),
        expected_dimensions=embedding_provider.dimensions,
        provider_name=dense_provider_name,
        model_name=dense_model_name,
    )
    emit_event(
        event_sink,
        EmbedCompleted(
            provider=dense_provider_name,
            model=dense_model_name,
            text_count=len(texts.dense_texts),
            role=DENSE_RETRIEVAL_CHANNEL,
            duration_ms=now_ms() - dense_started_ms,
            cache_hits=dense_cache.hits,
            cache_misses=dense_cache.misses,
            cache_writes=dense_cache.writes,
            cache_bypasses=dense_cache.bypasses,
        ),
    )

    sparse_provider_name = provider_name(sparse_embedder)
    sparse_model_name = getattr(sparse_embedder, "model_name", "")
    emit_event(
        event_sink,
        EmbedRequested(
            provider=sparse_provider_name,
            model=sparse_model_name,
            text_count=len(texts.sparse_texts),
            role=SPARSE_RETRIEVAL_CHANNEL,
        ),
    )
    sparse_started_ms = now_ms()
    sparse_channels = embed_sparse_channels(
        sparse_embedder=sparse_embedder,
        texts=texts.sparse_texts,
        expected_count=len(texts.chunks),
    )
    emit_event(
        event_sink,
        EmbedCompleted(
            provider=sparse_provider_name,
            model=sparse_model_name,
            text_count=len(texts.sparse_texts),
            role=SPARSE_RETRIEVAL_CHANNEL,
            duration_ms=now_ms() - sparse_started_ms,
        ),
    )

    return PreparedIndexData(
        content_type=texts.content_type,
        chunks=texts.chunks,
        dense_vectors=dense_vectors,
        payload_texts=texts.payload_texts,
        sparse_texts=texts.sparse_texts,
        sparse_channels=sparse_channels,
    )
