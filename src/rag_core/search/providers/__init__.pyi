from __future__ import annotations

from typing import Any

from rag_core.search.policy import VectorStorePolicy as VectorStorePolicy
from rag_core.search.providers.chunk_context_cache import (
    ChunkContextCache as ChunkContextCache,
    ChunkContextKey as ChunkContextKey,
)
from rag_core.search.providers.embedding_cache_models import (
    EmbedCacheKey as EmbedCacheKey,
    EmbeddingCache as EmbeddingCache,
)
from rag_core.search.providers.qdrant_store import QdrantVectorStore as QdrantVectorStore
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
from rag_core.search.types import (
    EmbeddingProvider,
    QueryPlanCapabilities as QueryPlanCapabilities,
    RerankerProvider,
    SearchSidecar,
    SparseEmbedder,
    StoreCapabilities as StoreCapabilities,
)

__all__: tuple[str, ...]

def create_chunk_context_cache(
    provider: str | None = ...,
    **kwargs: Any,
) -> ChunkContextCache: ...
def create_embedding_cache(
    provider: str | None = ...,
    **kwargs: Any,
) -> EmbeddingCache: ...
def create_embedding_provider(
    *,
    provider: str = ...,
    **kwargs: Any,
) -> EmbeddingProvider: ...
def create_reranker(
    provider: str = ...,
    model: str | None = ...,
    api_key: str | None = ...,
) -> RerankerProvider: ...
def create_search_sidecar(
    provider: str | None = ...,
    **kwargs: Any,
) -> SearchSidecar | None: ...
def create_sparse_embedder(
    *,
    provider: str = ...,
    **kwargs: Any,
) -> SparseEmbedder: ...
