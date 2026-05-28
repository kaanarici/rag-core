from __future__ import annotations

from rag_core.search.provider_protocols import EmbeddingProvider, VectorStore


def validate_index_namespace(namespace: str) -> str:
    namespace_scoped = namespace.strip()
    if not namespace_scoped:
        raise ValueError("namespace is required for indexing")
    return namespace_scoped


def validate_delete_scope(namespace: str, corpus_id: str) -> tuple[str, str]:
    namespace_scoped = namespace.strip()
    if not namespace_scoped:
        raise ValueError("namespace is required for delete_document")
    corpus_scoped = corpus_id.strip()
    if not corpus_scoped:
        raise ValueError("corpus_id is required for delete_document")
    return namespace_scoped, corpus_scoped


def validate_embedding_store_dimensions(
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
) -> None:
    store_dimensions = _store_dense_dimensions(vector_store)
    if store_dimensions is None:
        return
    embedding_dimensions = embedding_provider.dimensions
    if embedding_dimensions == store_dimensions:
        return
    raise ValueError(
        "Dense dimension mismatch before vector write: embedding provider "
        "produces %d dimensions, but vector store expects %d. Configure matching "
        "embedding dimensions or vector store dense_vector_dimensions."
        % (embedding_dimensions, store_dimensions)
    )


def validate_embedding_batch_size(batch_size: int) -> int:
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("embedding_batch_size must be a positive integer")
    return batch_size


def _store_dense_dimensions(vector_store: VectorStore) -> int | None:
    value = vector_store.capabilities.dense_vector_dimensions
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("vector store dense_vector_dimensions must be an integer")
    if value <= 0:
        raise ValueError("vector store dense_vector_dimensions must be positive")
    return value
