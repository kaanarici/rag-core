from __future__ import annotations

import logging

from rag_core.search.embedding_cache_diagnostics import EmbeddingCacheCounters
from rag_core.search.embedding_cache_diagnostics import (
    embed_texts_with_cache_observation,
)
from rag_core.search.provider_protocols import (
    EmbeddingProvider,
    SparseEmbedder,
    provider_name as _provider_name,
)
from rag_core.search.sparse_channels import single_sparse_channel
from rag_core.search.vector_models import SparseVector

logger = logging.getLogger(__name__)


async def embed_dense_texts(
    embedding_provider: EmbeddingProvider,
    texts: list[str],
    *,
    processing_fingerprint: str,
    batch_size: int,
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
            _provider_name(sparse_embedder),
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
