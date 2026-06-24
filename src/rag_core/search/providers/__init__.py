from __future__ import annotations

from importlib import import_module

__all__ = (
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

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ChunkContextCache": (
        "rag_core.search.providers.chunk_context_cache",
        "ChunkContextCache",
    ),
    "ChunkContextKey": (
        "rag_core.search.providers.chunk_context_cache",
        "ChunkContextKey",
    ),
    "EmbedCacheKey": (
        "rag_core.search.providers.embedding_cache_models",
        "EmbedCacheKey",
    ),
    "EmbeddingCache": (
        "rag_core.search.providers.embedding_cache_models",
        "EmbeddingCache",
    ),
    "QdrantVectorStore": (
        "rag_core.search.providers.qdrant_store",
        "QdrantVectorStore",
    ),
    "PgVectorVectorStore": (
        "rag_core.search.providers.pgvector_store",
        "PgVectorVectorStore",
    ),
    "VECTOR_STORES": ("rag_core.search.providers.registry", "VECTOR_STORES"),
    "create_chunk_context_cache": (
        "rag_core.search.providers.chunk_context_cache",
        "create_chunk_context_cache",
    ),
    "create_embedding_cache": (
        "rag_core.search.providers.embedding_cache",
        "create_embedding_cache",
    ),
    "create_embedding_provider": (
        "rag_core.search.providers.embedding",
        "create_embedding_provider",
    ),
    "create_reranker": ("rag_core.search.providers.reranker", "create_reranker"),
    "create_search_sidecar": (
        "rag_core.search.lexical_sidecar",
        "create_search_sidecar",
    ),
    "create_sparse_embedder": (
        "rag_core.search.providers.sparse",
        "create_sparse_embedder",
    ),
    "CHUNK_CONTEXT_CACHES": (
        "rag_core.search.providers.registry",
        "CHUNK_CONTEXT_CACHES",
    ),
    "EMBEDDING_CACHES": (
        "rag_core.search.providers.registry",
        "EMBEDDING_CACHES",
    ),
    "EMBEDDING_PROVIDERS": (
        "rag_core.search.providers.registry",
        "EMBEDDING_PROVIDERS",
    ),
    "RERANKER_PROVIDERS": ("rag_core.search.providers.registry", "RERANKER_PROVIDERS"),
    "SEARCH_SIDECARS": (
        "rag_core.search.providers.registry",
        "SEARCH_SIDECARS",
    ),
    "SPARSE_EMBEDDERS": ("rag_core.search.providers.registry", "SPARSE_EMBEDDERS"),
    "ProviderRegistry": ("rag_core.search.providers.registry", "ProviderRegistry"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, symbol = target
    return getattr(import_module(module_name), symbol)


def __dir__() -> list[str]:
    return list(__all__)
