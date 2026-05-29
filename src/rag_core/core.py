from __future__ import annotations

import inspect
from types import TracebackType
from typing import TYPE_CHECKING

from rag_core._engine.core_assembly import build_core_components
from rag_core.facade.ingest import _RAGCoreIngestMethods
from rag_core.facade.manifest import _RAGCoreManifestMethods
from rag_core.facade.prepare import _RAGCorePrepareMethods
from rag_core.facade.retrieval import _RAGCoreRetrievalMethods
from rag_core.core_models import RAGCoreConfig
from rag_core._engine.core_runtime import build_runtime_description

if TYPE_CHECKING:
    from rag_core._engine.core_ingest import CoreIngestor
    from rag_core.documents.contextualizer import ChunkContextualizer
    from rag_core.documents.ocr import OcrProvider
    from rag_core.events.sink import EventSink
    from rag_core.search.indexer import DocumentIndexer
    from rag_core.search.providers.chunk_context_cache import ChunkContextCache
    from rag_core.search.providers.embedding_cache_models import EmbeddingCache
    from rag_core.search.pipeline_runner import SearchPipelineRunner
    from rag_core.search.provider_protocols import (
        EmbeddingProvider,
        RerankerProvider,
        SearchSidecar,
        SparseEmbedder,
        VectorStore,
    )


class RAGCore(
    _RAGCorePrepareMethods,
    _RAGCoreIngestMethods,
    _RAGCoreManifestMethods,
    _RAGCoreRetrievalMethods,
):
    _config: RAGCoreConfig
    _ocr: "OcrProvider | None"
    _event_sink: "EventSink | None"
    _chunk_contextualizer: "ChunkContextualizer | None"
    _chunk_context_cache: "ChunkContextCache | None"
    _embedding_cache: "EmbeddingCache | None"
    _embedding: "EmbeddingProvider"
    _sparse: "SparseEmbedder"
    _store: "VectorStore"
    _reranker: "RerankerProvider | None"
    _sidecar: "SearchSidecar | None"
    _indexer: "DocumentIndexer"
    _search: "SearchPipelineRunner"
    _ingest: "CoreIngestor"
    _collection_name: str

    def __init__(
        self,
        config: RAGCoreConfig,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        sparse_embedder: SparseEmbedder | None = None,
        vector_store: VectorStore | None = None,
        reranker: RerankerProvider | None = None,
        ocr_provider: OcrProvider | None = None,
        search_sidecar: SearchSidecar | None = None,
        event_sink: "EventSink | None" = None,
        chunk_contextualizer: ChunkContextualizer | None = None,
        chunk_context_cache: ChunkContextCache | None = None,
        embedding_cache: EmbeddingCache | None = None,
    ) -> None:
        self._config = config
        self._ocr = ocr_provider
        self._event_sink = event_sink
        self._chunk_contextualizer = chunk_contextualizer
        components = build_core_components(
            config,
            embedding_provider=embedding_provider,
            sparse_embedder=sparse_embedder,
            vector_store=vector_store,
            reranker=reranker,
            search_sidecar=search_sidecar,
            prepare_bytes=self.prepare_bytes,
            event_sink=event_sink,
            embedding_cache=embedding_cache,
            chunk_contextualizer=chunk_contextualizer,
        )
        self._embedding = components.embedding
        self._sparse = components.sparse
        self._store = components.store
        self._reranker = components.reranker
        self._sidecar = components.sidecar
        self._indexer = components.indexer
        self._search = components.search
        self._ingest = components.ingest
        self._collection_name = components.collection_name
        self._processing_version = components.processing_version
        self._embedding_cache = components.embedding_cache
        self._chunk_context_cache = chunk_context_cache

    async def __aenter__(self) -> RAGCore:
        await self.ensure_ready()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def ensure_ready(self) -> None:
        await self._store.ensure_collection()

    async def check_health(self) -> dict[str, object]:
        return await self._store.check_health()

    async def close(self) -> None:
        close_errors = []
        for resource_name, resource in (
            ("vector_store", self._store),
            ("embedding_cache", self._embedding_cache),
            ("chunk_context_cache", self._chunk_context_cache),
        ):
            error = await _close_optional_resource(resource, resource_name=resource_name)
            if error is not None:
                close_errors.append(error)
        if not close_errors:
            return
        if len(close_errors) == 1:
            raise close_errors[0]
        raise ExceptionGroup("Failed to close RAGCore resources", close_errors)

    def describe_runtime(self) -> dict[str, object]:
        return build_runtime_description(
            config=self._config,
            collection_name=self._collection_name,
            embedding_provider=self._embedding,
            sparse_embedder=self._sparse,
            vector_store=self._store,
            reranker=self._reranker,
            ocr_provider=self._ocr,
            processing_version=self._processing_version,
            search_sidecar=self._sidecar,
            event_sink=self._event_sink,
            chunk_contextualizer=self._chunk_contextualizer,
            chunk_context_cache=self._chunk_context_cache,
            embedding_cache=self._embedding_cache,
        )


__all__ = ["RAGCore"]


async def _close_optional_resource(
    resource: object | None, *, resource_name: str
) -> Exception | None:
    if resource is None:
        return None
    close = getattr(resource, "close", None)
    if close is None:
        return None
    try:
        result = close()
        if inspect.isawaitable(result):
            await result
    except Exception as exc:
        exc.add_note(f"while closing RAGCore resource: {resource_name}")
        return exc
    return None
