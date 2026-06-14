from __future__ import annotations

from typing import Any

from rag_core.search.providers.chunk_context_cache import (
    ChunkContextCache as ChunkContextCache,
    ChunkContextKey as ChunkContextKey,
    create_chunk_context_cache as create_chunk_context_cache,
)
from rag_core.search.providers.embedding_cache_models import (
    EmbedCacheKey as EmbedCacheKey,
    EmbeddingCache as EmbeddingCache,
)
from rag_core.search.providers.qdrant_store import (
    QdrantVectorStore as QdrantVectorStore,
)
from rag_core.search.providers.pgvector_store import (
    PgVectorVectorStore as PgVectorVectorStore,
)
from rag_core.search.providers.registry import (
    CHUNK_CONTEXT_CACHES as CHUNK_CONTEXT_CACHES,
    EMBEDDING_CACHES as EMBEDDING_CACHES,
    EMBEDDING_PROVIDERS as EMBEDDING_PROVIDERS,
    RERANKER_PROVIDERS as RERANKER_PROVIDERS,
    SEARCH_SIDECARS as SEARCH_SIDECARS,
    SPARSE_EMBEDDERS as SPARSE_EMBEDDERS,
    VECTOR_STORES as VECTOR_STORES,
    ProviderRegistry as ProviderRegistry,
)
from rag_core.search.provider_protocols import (
    EmbeddingProvider as _EmbeddingProvider,
    RerankerProvider as _RerankerProvider,
    SearchSidecar as _SearchSidecar,
    SparseEmbedder as _SparseEmbedder,
)

__all__: tuple[str, ...] = (
    "CHUNK_CONTEXT_CACHES",
    "ChunkContextCache",
    "ChunkContextKey",
    "EMBEDDING_CACHES",
    "EMBEDDING_PROVIDERS",
    "EmbedCacheKey",
    "EmbeddingCache",
    "ProviderRegistry",
    "PgVectorVectorStore",
    "QdrantVectorStore",
    "RERANKER_PROVIDERS",
    "SEARCH_SIDECARS",
    "SPARSE_EMBEDDERS",
    "VECTOR_STORES",
    "create_chunk_context_cache",
    "create_embedding_cache",
    "create_embedding_provider",
    "create_reranker",
    "create_search_sidecar",
    "create_sparse_embedder",
)

def create_embedding_cache(
    provider: str | None = ...,
    **kwargs: Any,
) -> EmbeddingCache: ...
def create_embedding_provider(
    *,
    provider: str = ...,
    **kwargs: Any,
) -> _EmbeddingProvider: ...
def create_reranker(
    provider: str = ...,
    model: str | None = ...,
    api_key: str | None = ...,
) -> _RerankerProvider: ...
def create_search_sidecar(
    provider: str | None = ...,
    **kwargs: Any,
) -> _SearchSidecar | None: ...
def create_sparse_embedder(
    *,
    provider: str = ...,
    **kwargs: Any,
) -> _SparseEmbedder: ...
