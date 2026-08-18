from __future__ import annotations

import inspect
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

from rag_core._engine.core_assembly import build_core_components
from rag_core._engine.core_prepare import parse_document_bytes, prepare_document_bytes
from rag_core._engine.core_retrieval import context_with_core, search_with_core
from rag_core._engine.core_runtime import build_runtime_description
from rag_core.config.ingest_config import DEFAULT_INGEST_MAX_CONCURRENCY
from rag_core.core_models import (
    CollectionManifest,
    CollectionManifestEntry,
    Config,
    DeleteDocumentResult,
    IngestedDocument,
    ParsedDocument,
    PreparedDocument,
)
from rag_core.events.types import AuditContext
from rag_core.facade.ingest_batches import (
    ingest_archive_from_facade,
    ingest_files_from_facade,
    ingest_urls_from_facade,
)
from rag_core.facade.ingest_sources import (
    ingest_local_file_source,
    ingest_remote_url_source,
)
from rag_core.file_io import detect_local_mime_type, read_file_bytes
from rag_core.retrieval_defaults import (
    DEFAULT_CONTEXT_LIMIT,
    DEFAULT_RERANK,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_USE_LEXICAL_SEARCH,
)
from rag_core.scope import normalize_collection, normalize_namespace, resolve_collections_argument
from rag_core.search import Context, RerankBudget, SearchResult

if TYPE_CHECKING:
    from rag_core._engine.core_ingest import CoreIngestor
    from rag_core.documents.contextualizer import ChunkContextualizer
    from rag_core.documents.ocr import OcrProvider
    from rag_core.events.sink import EventSink
    from rag_core.fetch_security import FetchLimits, FetchSecurityPolicy
    from rag_core.fetching import FetchClient
    from rag_core.ingest.local.models import LocalIngestResult
    from rag_core.ingest.sources.archive import ArchiveLimits
    from rag_core.ingest.urls.models import RemoteUrlIngestResult
    from rag_core.search import Filter, QueryPlan
    from rag_core.search.indexer import DocumentIndexer
    from rag_core.search.pipeline_runner import SearchPipelineRunner
    from rag_core.search.provider_protocols import (
        EmbeddingProvider,
        RerankerProvider,
        SearchSidecar,
        SparseEmbedder,
        VectorStore,
    )
    from rag_core.search.providers.chunk_context_cache import ChunkContextCache
    from rag_core.search.providers.embedding_cache_models import EmbeddingCache


class Engine:
    def __init__(
        self,
        config: Config,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        sparse_embedder: SparseEmbedder | None = None,
        vector_store: VectorStore | None = None,
        reranker: RerankerProvider | None = None,
        ocr_provider: OcrProvider | None = None,
        search_sidecar: SearchSidecar | None = None,
        event_sink: EventSink | None = None,
        chunk_contextualizer: ChunkContextualizer | None = None,
        chunk_context_cache: ChunkContextCache | None = None,
        embedding_cache: EmbeddingCache | None = None,
    ) -> None:
        self._config = config
        self._ocr = ocr_provider
        self._event_sink = event_sink
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
            chunk_context_cache=chunk_context_cache,
        )
        self._embedding = components.embedding
        self._sparse = components.sparse
        self._store = components.store
        self._reranker = components.reranker
        self._sidecar = components.sidecar
        self._indexer: DocumentIndexer = components.indexer
        self._search: SearchPipelineRunner = components.search
        self._ingest: CoreIngestor = components.ingest
        self._collection_name = components.collection_name
        self._processing_version = components.processing_version
        self._embedding_cache = components.embedding_cache
        self._chunk_context_cache = components.chunk_context_cache
        self._chunk_contextualizer = components.chunk_contextualizer

    async def __aenter__(self) -> Engine:
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
        raise ExceptionGroup("Failed to close Engine resources", close_errors)

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

    def describe_event_sink_status(self) -> dict[str, object]:
        from rag_core.events.sinks import describe_event_sink_status as _describe

        return _describe(self._event_sink)

    async def parse_bytes(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        path: str | None = None,
    ) -> ParsedDocument:
        return await parse_document_bytes(
            file_bytes=file_bytes,
            filename=filename,
            mime_type=mime_type,
            path=path,
            event_sink=self._event_sink,
        )

    async def prepare_bytes(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        path: str | None = None,
        namespace: str = "",
        collection: str = "",
        document_id: str = "",
    ) -> PreparedDocument:
        return await prepare_document_bytes(
            file_bytes=file_bytes,
            filename=filename,
            mime_type=mime_type,
            path=path,
            namespace=namespace,
            collection=collection,
            document_id=document_id,
            ocr_provider=self._ocr,
            event_sink=self._event_sink,
            contextualizer=self._chunk_contextualizer,
            chunk_context_cache=self._chunk_context_cache,
            chunking_config=self._config.chunking,
        )

    async def prepare_file(
        self,
        path: str | Path,
        *,
        mime_type: str | None = None,
    ) -> PreparedDocument:
        file_path = Path(path)
        return await self.prepare_bytes(
            file_bytes=await read_file_bytes(file_path),
            filename=file_path.name,
            mime_type=mime_type or detect_local_mime_type(file_path),
            path=str(file_path),
        )

    async def add_bytes(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        collection: str | None = None,
        namespace: str | None = None,
        document_id: str | None = None,
        document_key: str | None = None,
        path: str | None = None,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        source_type: str | None = None,
        audit_context: AuditContext | None = None,
        ingest_id: str | None = None,
    ) -> IngestedDocument:
        return await self._ingest.ingest_bytes(
            file_bytes=file_bytes,
            filename=filename,
            mime_type=mime_type,
            namespace=normalize_namespace(namespace),
            collection=normalize_collection(collection),
            document_id=document_id,
            document_key=document_key,
            path=path,
            metadata=metadata,
            force_reindex=force_reindex,
            source_type=source_type,
            audit_context=audit_context,
            ingest_id=ingest_id,
        )

    async def add_file(
        self,
        path: str | Path,
        *,
        collection: str | None = None,
        namespace: str | None = None,
        document_id: str | None = None,
        document_key: str | None = None,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        audit_context: AuditContext | None = None,
        ingest_id: str | None = None,
        pre_read_bytes: bytes | None = None,
    ) -> IngestedDocument:
        return await ingest_local_file_source(
            path,
            ingest_bytes=self.add_bytes,
            namespace=normalize_namespace(namespace),
            collection=normalize_collection(collection),
            document_id=document_id,
            document_key=document_key,
            metadata=metadata,
            force_reindex=force_reindex,
            audit_context=audit_context,
            ingest_id=ingest_id,
            pre_read_bytes=pre_read_bytes,
        )

    async def add(
        self,
        path: str | Path,
        *,
        collection: str | None = None,
        namespace: str | None = None,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        max_concurrency: int = DEFAULT_INGEST_MAX_CONCURRENCY,
        manifest_dir: str | Path | None = None,
    ) -> LocalIngestResult:
        return await ingest_files_from_facade(
            core=self,
            config=self._config,
            event_sink=self._event_sink,
            path=path,
            namespace=normalize_namespace(namespace),
            collection=normalize_collection(collection),
            metadata=metadata,
            force_reindex=force_reindex,
            max_concurrency=max_concurrency,
            manifest_dir=manifest_dir,
        )

    async def add_archive(
        self,
        archive_path: str | Path,
        *,
        collection: str | None = None,
        namespace: str | None = None,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        max_concurrency: int = DEFAULT_INGEST_MAX_CONCURRENCY,
        archive_limits: ArchiveLimits | None = None,
        manifest_dir: str | Path | None = None,
    ) -> LocalIngestResult:
        return await ingest_archive_from_facade(
            core=self,
            config=self._config,
            event_sink=self._event_sink,
            archive_path=archive_path,
            namespace=normalize_namespace(namespace),
            collection=normalize_collection(collection),
            metadata=metadata,
            force_reindex=force_reindex,
            max_concurrency=max_concurrency,
            archive_limits=archive_limits,
            manifest_dir=manifest_dir,
        )

    async def add_url(
        self,
        url: str,
        *,
        collection: str | None = None,
        namespace: str | None = None,
        document_id: str | None = None,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        fetch_client: FetchClient | None = None,
        fetch_policy: FetchSecurityPolicy | None = None,
        fetch_limits: FetchLimits | None = None,
    ) -> IngestedDocument:
        return await ingest_remote_url_source(
            url,
            ingest_bytes=self.add_bytes,
            namespace=normalize_namespace(namespace),
            collection=normalize_collection(collection),
            event_sink=self._event_sink,
            document_id=document_id,
            metadata=metadata,
            force_reindex=force_reindex,
            fetch_client=fetch_client,
            fetch_policy=fetch_policy,
            fetch_limits=fetch_limits,
        )

    async def add_urls(
        self,
        url_file: str | Path | None = None,
        *,
        urls: Sequence[str] | None = None,
        collection: str | None = None,
        namespace: str | None = None,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        max_concurrency: int = DEFAULT_INGEST_MAX_CONCURRENCY,
        fetch_client: FetchClient | None = None,
        fetch_policy: FetchSecurityPolicy | None = None,
        fetch_limits: FetchLimits | None = None,
        manifest_dir: str | Path | None = None,
    ) -> RemoteUrlIngestResult:
        return await ingest_urls_from_facade(
            core=self,
            config=self._config,
            event_sink=self._event_sink,
            url_file=url_file,
            urls=urls,
            namespace=normalize_namespace(namespace),
            collection=normalize_collection(collection),
            metadata=metadata,
            force_reindex=force_reindex,
            max_concurrency=max_concurrency,
            fetch_client=fetch_client,
            fetch_policy=fetch_policy,
            fetch_limits=fetch_limits,
            manifest_dir=manifest_dir,
        )

    async def delete_document(
        self,
        *,
        document_id: str,
        collection: str | None = None,
        namespace: str | None = None,
    ) -> DeleteDocumentResult:
        return await self._ingest.delete_document(
            document_id=document_id,
            namespace=normalize_namespace(namespace),
            collection=normalize_collection(collection),
        )

    async def delete_collection(
        self,
        *,
        collection: str | None = None,
        namespace: str | None = None,
    ) -> None:
        await self._ingest.delete_collection(
            namespace=normalize_namespace(namespace),
            collection=normalize_collection(collection),
        )

    async def delete_namespace(self, *, namespace: str) -> None:
        await self._ingest.delete_namespace(namespace=namespace)

    async def search(
        self,
        *,
        query: str,
        collection: str | None = None,
        collections: Sequence[str] | None = None,
        namespace: str | None = None,
        limit: int = DEFAULT_SEARCH_LIMIT,
        content_types: list[str] | None = None,
        document_ids: list[str] | None = None,
        rerank: bool = DEFAULT_RERANK,
        use_lexical_search: bool = DEFAULT_USE_LEXICAL_SEARCH,
        query_plan: QueryPlan | None = None,
        metadata_filter: Filter | None = None,
        rerank_budget: RerankBudget | None = None,
        audit_context: AuditContext | None = None,
    ) -> list[SearchResult]:
        return await search_with_core(
            search=self._search,
            query=query,
            namespace=normalize_namespace(namespace),
            collections=resolve_collections_argument(
                collection=collection,
                collections=collections,
                caller="Engine.search",
            ),
            limit=limit,
            content_types=content_types,
            document_ids=document_ids,
            rerank=rerank,
            use_lexical_search=use_lexical_search,
            query_plan=query_plan,
            metadata_filter=metadata_filter,
            rerank_budget=rerank_budget,
            audit_context=audit_context,
        )

    async def context(
        self,
        *,
        query: str,
        collection: str | None = None,
        collections: Sequence[str] | None = None,
        namespace: str | None = None,
        limit: int = DEFAULT_CONTEXT_LIMIT,
        content_types: list[str] | None = None,
        document_ids: list[str] | None = None,
        rerank: bool = DEFAULT_RERANK,
        use_lexical_search: bool = DEFAULT_USE_LEXICAL_SEARCH,
        query_plan: QueryPlan | None = None,
        metadata_filter: Filter | None = None,
        rerank_budget: RerankBudget | None = None,
        max_chars: int | None = None,
        max_tokens: int | None = None,
        audit_context: AuditContext | None = None,
    ) -> Context:
        return await context_with_core(
            search=self._search,
            event_sink=self._event_sink,
            query=query,
            namespace=normalize_namespace(namespace),
            collections=resolve_collections_argument(
                collection=collection,
                collections=collections,
                caller="Engine.context",
            ),
            limit=limit,
            content_types=content_types,
            document_ids=document_ids,
            rerank=rerank,
            use_lexical_search=use_lexical_search,
            query_plan=query_plan,
            metadata_filter=metadata_filter,
            rerank_budget=rerank_budget,
            max_chars=max_chars,
            max_tokens=max_tokens,
            audit_context=audit_context,
        )

    def build_manifest_entry(
        self,
        *,
        document: IngestedDocument,
    ) -> CollectionManifestEntry:
        from rag_core._engine.core_manifest import manifest_entry_for_core

        return manifest_entry_for_core(document)

    async def manifest_bytes(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        namespace: str,
        collection: str,
        document_id: str | None = None,
        document_key: str | None = None,
        path: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> CollectionManifestEntry:
        from rag_core._engine.core_manifest import manifest_bytes_for_core

        return await manifest_bytes_for_core(
            prepare_bytes=self.prepare_bytes,
            collection_name=self._collection_name,
            embedding_model=self._embedding.model_name,
            file_bytes=file_bytes,
            filename=filename,
            mime_type=mime_type,
            namespace=namespace,
            collection=collection,
            document_id=document_id,
            document_key=document_key,
            path=path,
            metadata=metadata,
        )

    async def manifest_file(
        self,
        path: str | Path,
        *,
        namespace: str,
        collection: str,
        document_id: str | None = None,
        document_key: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> CollectionManifestEntry:
        from rag_core._engine.core_manifest import manifest_file_for_core

        return await manifest_file_for_core(
            path,
            prepare_bytes=self.prepare_bytes,
            collection_name=self._collection_name,
            embedding_model=self._embedding.model_name,
            namespace=namespace,
            collection=collection,
            document_id=document_id,
            document_key=document_key,
            metadata=metadata,
        )

    def build_collection_manifest(
        self,
        *,
        namespace: str,
        collection: str,
        documents: list[IngestedDocument],
    ) -> CollectionManifest:
        from rag_core._engine.core_manifest import collection_manifest_for_core

        return collection_manifest_for_core(
            namespace=namespace,
            collection=collection,
            collection_name=self._collection_name,
            embedding_provider=self._config.embedding.provider,
            embedding_model=self._embedding.model_name,
            embedding_dimensions=self._embedding.dimensions,
            documents=documents,
        )


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
        exc.add_note(f"while closing Engine resource: {resource_name}")
        return exc
    return None


__all__ = ["Engine"]
