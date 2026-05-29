from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag_core._engine.core_ingest import CoreIngestor, PrepareBytes
from rag_core.core_models import ProcessingFingerprint, RAGCoreConfig
from rag_core._engine.core_runtime import (
    resolve_processing_version,
    resolve_runtime_collection_name,
)
from rag_core._engine.core_vector_store_factory import (
    create_configured_vector_store,
    require_vector_store_capabilities,
)
from rag_core.search.indexer import DocumentIndexer
from rag_core.search.lexical_sidecar import PortableLexicalSidecar
from rag_core.search.pipeline_runner import SearchPipelineRunner
from rag_core.search.providers.cache_provider_names import SQLITE_CACHE_PROVIDER

if TYPE_CHECKING:
    from rag_core.documents.contextualizer import ChunkContextualizer
    from rag_core.events.sink import EventSink
    from rag_core.search.providers.embedding_cache_models import EmbeddingCache
    from rag_core.search.provider_protocols import (
        EmbeddingProvider,
        RerankerProvider,
        SearchSidecar,
        SparseEmbedder,
        VectorStore,
    )


@dataclass(frozen=True)
class CoreComponents:
    embedding: EmbeddingProvider
    sparse: SparseEmbedder
    store: VectorStore
    reranker: RerankerProvider | None
    sidecar: SearchSidecar | None
    indexer: DocumentIndexer
    search: SearchPipelineRunner
    ingest: CoreIngestor
    collection_name: str
    processing_version: ProcessingFingerprint
    embedding_cache: EmbeddingCache | None


def build_core_components(
    config: RAGCoreConfig,
    *,
    embedding_provider: EmbeddingProvider | None,
    sparse_embedder: SparseEmbedder | None,
    vector_store: VectorStore | None,
    reranker: RerankerProvider | None,
    search_sidecar: SearchSidecar | None,
    prepare_bytes: PrepareBytes,
    event_sink: "EventSink | None" = None,
    embedding_cache: "EmbeddingCache | None" = None,
    chunk_contextualizer: "ChunkContextualizer | None" = None,
) -> CoreComponents:
    from rag_core.search.providers.embedding import create_embedding_provider
    from rag_core.search.providers.registry import VECTOR_STORES
    from rag_core.search.providers.reranker import create_reranker, is_noop_reranker
    from rag_core.search.providers.sparse import FastEmbedSparseEmbedder

    embedding = embedding_provider or create_embedding_provider(
        provider=config.embedding.provider,
        model=config.embedding.model,
        dimensions=config.embedding.dimensions,
        api_key=config.embedding.api_key,
        base_url=config.embedding.base_url,
    )
    processing_version = resolve_processing_version(
        configured_version=config.ingest.processing_version,
        source_type=config.ingest.source_type,
        contextualizer_id=(
            chunk_contextualizer.contextualizer_id
            if chunk_contextualizer is not None
            else None
        ),
    )
    resolved_embedding_cache = embedding_cache
    if resolved_embedding_cache is None and config.ingest.embedding_cache_provider:
        from rag_core.search.providers.embedding_cache import create_embedding_cache

        cache_kwargs = {}
        if config.ingest.embedding_cache_provider == SQLITE_CACHE_PROVIDER:
            if config.ingest.embedding_cache_path is None:
                raise ValueError(
                    "ingest.embedding_cache_path is required when "
                    f"embedding_cache_provider={SQLITE_CACHE_PROVIDER!r}"
                )
            cache_kwargs["path"] = config.ingest.embedding_cache_path
        resolved_embedding_cache = create_embedding_cache(
            config.ingest.embedding_cache_provider,
            **cache_kwargs,
        )
    if resolved_embedding_cache is not None:
        from rag_core.search.providers.cached_embedding import CachedEmbeddingProvider

        embedding = CachedEmbeddingProvider(
            embedding,
            resolved_embedding_cache,
            processing_fingerprint=processing_version.serialize(),
        )
    sparse = sparse_embedder or FastEmbedSparseEmbedder()
    collection_name = resolve_runtime_collection_name(
        config=config,
        model_name=embedding.model_name,
        dimensions=embedding.dimensions,
    )
    store = vector_store or create_configured_vector_store(
        config=config,
        collection_name=collection_name,
        dense_dimensions=embedding.dimensions,
        vector_stores=VECTOR_STORES,
    )
    require_vector_store_capabilities(store)
    resolved_reranker = reranker or create_reranker(
        provider=config.reranker.provider,
        model=config.reranker.model,
        api_key=config.reranker.api_key,
    )
    search_reranker = None if is_noop_reranker(resolved_reranker) else resolved_reranker
    sidecar = search_sidecar
    if sidecar is None and config.ingest.lexical_search_provider:
        from rag_core.search.lexical_sidecar import create_search_sidecar

        sidecar = create_search_sidecar(config.ingest.lexical_search_provider)
    if sidecar is None and config.ingest.enable_lexical_search:
        sidecar = PortableLexicalSidecar([])

    indexer = DocumentIndexer(
        embedding_provider=embedding,
        sparse_embedder=sparse,
        vector_store=store,
        event_sink=event_sink,
        policy=config.policy,
        embedding_batch_size=config.embedding.batch_size,
    )
    search = SearchPipelineRunner(
        embedding_provider=embedding,
        sparse_embedder=sparse,
        vector_store=store,
        reranker=search_reranker,
        sidecar=sidecar,
        event_sink=event_sink,
    )
    ingest = CoreIngestor(
        collection_name=collection_name,
        source_type=config.ingest.source_type,
        embedding_model=embedding.model_name,
        processing_version=processing_version,
        store=store,
        indexer=indexer,
        sidecar=sidecar,
        prepare_bytes=prepare_bytes,
        event_sink=event_sink,
        manifest_directory=config.ingest.manifest_directory,
        policy=config.policy,
        skip_unchanged=config.ingest.skip_unchanged,
    )
    return CoreComponents(
        embedding=embedding,
        sparse=sparse,
        store=store,
        reranker=search_reranker,
        sidecar=sidecar,
        indexer=indexer,
        search=search,
        ingest=ingest,
        collection_name=collection_name,
        processing_version=processing_version,
        embedding_cache=resolved_embedding_cache,
    )
