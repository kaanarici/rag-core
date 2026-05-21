from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag_core.config.embedding_config import DEFAULT_EMBEDDING_BATCH_SIZE
from rag_core.core_models import PreparedChunk
from rag_core.events.emit import emit_event, now_ms
from rag_core.events.types import EmbedCompleted, EmbedRequested
from rag_core.search.indexer_embedding_vectors import (
    embed_dense_texts as _embed_dense_texts,
    embed_sparse_channels as _embed_sparse_channels,
    provider_name as _provider_name,
    validate_dense_vectors as _validate_dense_vectors,
    validate_embedding_batch_size as _validate_embedding_batch_size,
)
from rag_core.search.types import (
    ContentType,
    EmbeddingProvider,
    SparseEmbedder,
    SparseVector,
)

from .indexer_models import IndexRequest
from .indexer_texts import build_index_texts

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink


@dataclass(frozen=True)
class PreparedIndexData:
    content_type: ContentType
    chunks: list[PreparedChunk]
    dense_vectors: list[list[float]]
    payload_texts: list[str]
    sparse_channels: list[dict[str, SparseVector]]


async def prepare_index_data(
    *,
    req: IndexRequest,
    embedding_provider: EmbeddingProvider,
    sparse_embedder: SparseEmbedder,
    event_sink: "EventSink | None" = None,
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
) -> PreparedIndexData:
    _validate_embedding_batch_size(embedding_batch_size)
    texts = build_index_texts(req)
    if not texts.chunks:
        return PreparedIndexData(
            content_type=texts.content_type,
            chunks=[],
            dense_vectors=[],
            payload_texts=[],
            sparse_channels=[],
        )

    dense_provider_name = _provider_name(embedding_provider)
    dense_model_name = getattr(embedding_provider, "model_name", "")
    emit_event(
        event_sink,
        EmbedRequested(
            provider=dense_provider_name,
            model=dense_model_name,
            text_count=len(texts.dense_texts),
            role="dense",
        ),
    )
    dense_started_ms = now_ms()
    dense_vectors, dense_cache = await _embed_dense_texts(
        embedding_provider,
        texts.dense_texts,
        processing_fingerprint=req.processing_version or "",
        batch_size=embedding_batch_size,
    )
    _validate_dense_vectors(
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
            role="dense",
            duration_ms=now_ms() - dense_started_ms,
            cache_hits=dense_cache.hits,
            cache_misses=dense_cache.misses,
            cache_writes=dense_cache.writes,
            cache_bypasses=dense_cache.bypasses,
        ),
    )

    sparse_provider_name = _provider_name(sparse_embedder)
    sparse_model_name = getattr(sparse_embedder, "model_name", "")
    emit_event(
        event_sink,
        EmbedRequested(
            provider=sparse_provider_name,
            model=sparse_model_name,
            text_count=len(texts.sparse_texts),
            role="sparse",
        ),
    )
    sparse_started_ms = now_ms()
    sparse_channels = _embed_sparse_channels(
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
            role="sparse",
            duration_ms=now_ms() - sparse_started_ms,
        ),
    )

    return PreparedIndexData(
        content_type=texts.content_type,
        chunks=texts.chunks,
        dense_vectors=dense_vectors,
        payload_texts=texts.payload_texts,
        sparse_channels=sparse_channels,
    )
