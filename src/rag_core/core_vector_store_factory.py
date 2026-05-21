from __future__ import annotations

from typing import TYPE_CHECKING

from rag_core.core_models import RAGCoreConfig

if TYPE_CHECKING:
    from rag_core.search.providers.registry import ProviderRegistry
    from rag_core.search.types import VectorStore


def create_configured_vector_store(
    *,
    config: RAGCoreConfig,
    collection_name: str,
    dense_dimensions: int,
    vector_stores: "ProviderRegistry[VectorStore]",
) -> "VectorStore":
    if config.vector_store.provider == "qdrant":
        return vector_stores.create(
            "qdrant",
            url=config.qdrant.url,
            location=config.qdrant.location,
            api_key=config.qdrant.api_key,
            collection_name=collection_name,
            dense_dimensions=dense_dimensions,
            policy=config.policy,
        )
    if config.vector_store.provider == "turbopuffer":
        tp = config.vector_store.turbopuffer
        return vector_stores.create(
            "turbopuffer",
            namespace=collection_name,
            dense_dimensions=dense_dimensions,
            api_key=tp.api_key,
            region=tp.region,
            base_url=tp.base_url,
            distance_metric=tp.distance_metric,
            delete_continuation_limit=tp.delete_continuation_limit,
            policy=config.policy,
        )
    raise ValueError(f"Unsupported vector store provider: {config.vector_store.provider}")


def require_vector_store_capabilities(store: "VectorStore") -> None:
    capabilities = store.capabilities
    if capabilities.document_record_lookup:
        return
    adapter_name = type(store).__name__
    raise ValueError(
        f"VectorStore adapter {adapter_name!r} does not declare "
        "document_record_lookup capability; ingest dedup cannot run "
        "against it. Use an adapter that supports document record "
        "lookup (e.g. QdrantVectorStore, TurboPufferVectorStore, "
        "InMemoryVectorStore) or "
        "extend the adapter to declare the capability."
    )
