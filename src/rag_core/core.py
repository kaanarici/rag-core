from __future__ import annotations
from rag_core.search.request_models import SearchMode

from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from dataclasses import replace
from typing import TYPE_CHECKING

from rag_core.ingest.archive import ingest_zip_archive_with_core
from rag_core._engine.runtime_operations import RuntimeOperations, operation
from rag_core._engine.core_prepare import parse_document_bytes, prepare_document_bytes
from rag_core._engine.core_remote import ingest_remote_url
from rag_core._engine.core_runtime import build_runtime_description
from rag_core.config.ingest_config import DEFAULT_INGEST_MAX_CONCURRENCY
from rag_core.core_models import (
    Config,
    DeleteDocumentResult,
    IngestedDocument,
    ParsedDocument,
    PreparedDocument,
)
from rag_core.events.types import AuditContext
from rag_core.search.policy import validate_ingest_scope
from rag_core.file_io import detect_local_mime_type, read_file_bytes, validate_document_size
from rag_core.ingest.local import LocalIngestRequest, run_local_ingest_with_core
from rag_core.ingest.sources.local import (
    ingest_local_file_source,
    reject_local_symlink_path,
    reject_local_hardlink_path,
)
from rag_core.retrieval_defaults import (
    DEFAULT_CONTEXT_LIMIT,
    DEFAULT_RERANK,
    DEFAULT_SEARCH_LIMIT,
)
from rag_core.scope import (
    normalize_collection,
    normalize_tenant_id,
    resolve_collections_argument,
)
from rag_core.search import Context, RerankBudget, SearchResult

if TYPE_CHECKING:
    from rag_core.documents.contextualizer import ChunkContextualizer
    from rag_core.documents.ocr import OcrProvider
    from rag_core.events.sink import EventSink
    from rag_core.fetch_security import FetchLimits, FetchSecurityPolicy
    from rag_core.fetching import FetchClient
    from rag_core.ingest.local import LocalIngestResult
    from rag_core.ingest.sources.archive import ArchiveLimits
    from rag_core.search import (
        Filter,
    )
    from rag_core.search.provider_protocols import (
        EmbeddingProvider,
        RerankerProvider,
        SparseEmbedder,
        VectorStore,
    )


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
        event_sink: EventSink | None = None,
        chunk_contextualizer: ChunkContextualizer | None = None,
        max_concurrent_ingests: int = 8,
        max_concurrent_searches: int = 32,
        max_pending_operations: int = 64,
    ) -> None:
        self._operations = RuntimeOperations(
            max_concurrent_ingests=max_concurrent_ingests,
            max_concurrent_searches=max_concurrent_searches,
            max_pending_operations=max_pending_operations,
        )
        self._config = config
        self._ocr = ocr_provider
        self._event_sink = event_sink
        from rag_core._engine.core_runtime import (
            resolve_processing_version,
            resolve_runtime_collection_name,
        )
        from rag_core.search.providers.embedding import create_embedding_provider
        from rag_core.search.providers.sparse import FastEmbedSparseEmbedder
        from rag_core.search.providers.qdrant_store import QdrantVectorStore
        from rag_core.search.indexer import DocumentIndexer
        from rag_core.search.runner import SearchRunner
        from rag_core.ingest.document import CoreIngestor
        from rag_core.documents.contextualizer import contextualizer_chunk_cap

        self._embedding = embedding_provider or create_embedding_provider(
            provider=config.embedding.provider,
            model=config.embedding.model,
            dimensions=config.embedding.dimensions,
            api_key=config.embedding.api_key,
            base_url=config.embedding.base_url,
        )
        self._sparse = sparse_embedder
        native_lexical = (
            vector_store is not None and vector_store.capabilities.native_lexical
        )
        if (
            self._sparse is None
            and (config.hybrid or config.ingest.build_sparse_index)
            and not native_lexical
        ):
            self._sparse = FastEmbedSparseEmbedder()
        sparse_model = None
        if self._sparse is not None:
            sparse_model = getattr(self._sparse, "model_name", None)
            if not isinstance(sparse_model, str) or not sparse_model.strip():
                raise ValueError("SparseEmbedder.model_name must identify its model/vocabulary")
        self._collection_name = resolve_runtime_collection_name(
            config=config,
            model_name=self._embedding.model_name,
            dimensions=self._embedding.dimensions,
        )
        self._processing_version = resolve_processing_version(
            configured_version=config.ingest.processing_version,
            source_type=config.ingest.source_type,
            contextualizer_id=chunk_contextualizer.contextualizer_id
            if chunk_contextualizer
            else None,
            contextualizer_chunk_cap=contextualizer_chunk_cap(chunk_contextualizer),
            chunking=config.chunking,
            sparse_index=self._sparse is not None or native_lexical,
            sparse_model=sparse_model,
        )
        target = config.vector_store.qdrant
        self._store = vector_store or QdrantVectorStore(
            url=target.url,
            location=target.location,
            api_key=target.api_key,
            collection_name=self._collection_name,
            dense_dimensions=self._embedding.dimensions,
            sparse_enabled=self._sparse is not None,
            policy=config.policy,
            embedding_model=self._embedding.model_name,
        )
        self._reranker = reranker
        self._chunk_contextualizer = chunk_contextualizer
        self._indexer = DocumentIndexer(
            embedding_provider=self._embedding,
            sparse_embedder=self._sparse,
            vector_store=self._store,
            event_sink=event_sink,
            policy=config.policy,
            embedding_batch_size=config.embedding.batch_size,
            collection_policy=config.collection_policy,
        )
        self._search = SearchRunner(
            embedding_provider=self._embedding,
            sparse_embedder=self._sparse,
            vector_store=self._store,
            reranker=reranker,
            event_sink=event_sink,
            collection_policy=config.collection_policy,
        )
        self._ingest = CoreIngestor(
            collection_name=self._collection_name,
            source_type=config.ingest.source_type,
            embedding_model=self._embedding.model_name,
            embedding_dimensions=self._embedding.dimensions,
            ocr_identity=(self._ocr.provider_name, self._ocr.model_name)
            if self._ocr
            else None,
            processing_version=self._processing_version,
            store=self._store,
            indexer=self._indexer,
            prepare_bytes=self._prepare_bytes,
            event_sink=event_sink,
            manifest_directory=config.ingest.manifest_directory,
            policy=config.policy,
            skip_unchanged=config.ingest.skip_unchanged,
            collection_policy=config.collection_policy,
        )

    async def __aenter__(self) -> Engine:
        try:
            await self.ensure_ready()
        except BaseException:
            try:
                await self.close()
            except Exception:
                pass
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def _ensure_ready(self) -> None:
        await self._store.ensure_collection()

    @operation()
    async def check_health(self) -> dict[str, object]:
        return await self._store.check_health()

    async def close(self) -> None:
        """Stop admission and drain work; call outside Engine-owned callbacks."""
        await self._operations.close(self._store.close)

    def describe_runtime(self) -> dict[str, object]:
        description = build_runtime_description(
            config=self._config,
            collection_name=self._collection_name,
            embedding_provider=self._embedding,
            sparse_embedder=self._sparse,
            vector_store=self._store,
            reranker=self._reranker,
            ocr_provider=self._ocr,
            processing_version=self._processing_version,
            event_sink=self._event_sink,
            chunk_contextualizer=self._chunk_contextualizer,
        )
        description["operations"] = self._operations.snapshot()
        return description

    def describe_event_sink_status(self) -> dict[str, object]:
        from rag_core.events.sinks import describe_event_sink_status as _describe

        return _describe(self._event_sink)

    @operation(ingest=True)
    async def parse_bytes(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        path: str | None = None,
    ) -> ParsedDocument:
        validate_document_size(file_bytes, max_bytes=self._config.ingest.max_document_bytes)
        return await parse_document_bytes(
            file_bytes=file_bytes,
            filename=filename,
            mime_type=mime_type,
            path=path,
            event_sink=self._event_sink,
        )

    async def _prepare_bytes(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        path: str | None = None,
        tenant_id: str = "",
        collection: str = "",
        document_id: str = "",
        event_sink: EventSink | None = None,
    ) -> PreparedDocument:
        validate_document_size(file_bytes, max_bytes=self._config.ingest.max_document_bytes)
        return await prepare_document_bytes(
            file_bytes=file_bytes,
            filename=filename,
            mime_type=mime_type,
            path=path,
            tenant_id=tenant_id,
            collection=collection,
            document_id=document_id,
            ocr_provider=self._ocr,
            event_sink=event_sink if event_sink is not None else self._event_sink,
            contextualizer=self._chunk_contextualizer,
            chunking_config=self._config.chunking,
        )

    @operation(ingest=True)
    async def prepare_file(
        self,
        path: str | Path,
        *,
        mime_type: str | None = None,
    ) -> PreparedDocument:
        file_path = Path(path)
        reject_local_symlink_path(file_path)
        reject_local_hardlink_path(file_path)
        return await self._prepare_bytes(
            file_bytes=await read_file_bytes(
                file_path, max_bytes=self._config.ingest.max_document_bytes
            ),
            filename=file_path.name,
            mime_type=mime_type or detect_local_mime_type(file_path),
            path=str(file_path),
        )

    async def _add_bytes(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        collection: str | None = None,
        tenant_id: str | None = None,
        document_id: str | None = None,
        document_key: str | None = None,
        path: str | None = None,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        source_type: str | None = None,
        audit_context: AuditContext | None = None,
        ingest_id: str | None = None,
    ) -> IngestedDocument:
        tenant_id, collection = validate_ingest_scope(
            tenant_id, collection, self._config.collection_policy
        )
        validate_document_size(file_bytes, max_bytes=self._config.ingest.max_document_bytes)
        return await self._ingest.ingest_bytes(
            file_bytes=file_bytes,
            filename=filename,
            mime_type=mime_type,
            tenant_id=normalize_tenant_id(tenant_id),
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

    async def _add_file(
        self,
        path: str | Path,
        *,
        collection: str | None = None,
        tenant_id: str | None = None,
        document_id: str | None = None,
        document_key: str | None = None,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        audit_context: AuditContext | None = None,
        ingest_id: str | None = None,
        pre_read_bytes: bytes | None = None,
    ) -> IngestedDocument:
        tenant_id, collection = validate_ingest_scope(
            tenant_id, collection, self._config.collection_policy
        )
        return await ingest_local_file_source(
            path,
            ingest_bytes=self._add_bytes,
            tenant_id=normalize_tenant_id(tenant_id),
            collection=normalize_collection(collection),
            document_id=document_id,
            document_key=document_key,
            metadata=metadata,
            force_reindex=force_reindex,
            audit_context=audit_context,
            ingest_id=ingest_id,
            pre_read_bytes=pre_read_bytes,
            max_document_bytes=self._config.ingest.max_document_bytes,
        )

    @operation(batch=True)
    async def add(
        self,
        path: str | Path,
        *,
        collection: str | None = None,
        tenant_id: str | None = None,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        max_concurrency: int = DEFAULT_INGEST_MAX_CONCURRENCY,
        max_files: int | None = None,
        prune_missing: bool = False,
    ) -> LocalIngestResult:
        tenant_id, collection = validate_ingest_scope(
            tenant_id, collection, self._config.collection_policy
        )
        return await run_local_ingest_with_core(
            LocalIngestRequest(
                path=path,
                tenant_id=normalize_tenant_id(tenant_id),
                collection=normalize_collection(collection),
                metadata=metadata,
                force_reindex=force_reindex,
                max_concurrency=max_concurrency,
                max_files=max_files,
                prune_missing=prune_missing,
            ),
            core=_IngestCallbacks(self),
            event_sink=self._event_sink,
            manifest_dir=self._config.ingest.manifest_directory,
            max_document_bytes=self._config.ingest.max_document_bytes,
        )

    @operation(batch=True)
    async def add_archive(
        self,
        archive_path: str | Path,
        *,
        collection: str | None = None,
        tenant_id: str | None = None,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        max_concurrency: int = DEFAULT_INGEST_MAX_CONCURRENCY,
        archive_limits: ArchiveLimits | None = None,
    ) -> LocalIngestResult:
        tenant_id, collection = validate_ingest_scope(
            tenant_id, collection, self._config.collection_policy
        )
        from rag_core.ingest.sources.archive import ArchiveLimits

        archive_limits = archive_limits or ArchiveLimits()
        archive_limits = replace(
            archive_limits,
            max_entry_bytes=min(
                archive_limits.max_entry_bytes, self._config.ingest.max_document_bytes
            ),
        )
        return await ingest_zip_archive_with_core(
            core=_IngestCallbacks(self),
            archive_path=archive_path,
            tenant_id=normalize_tenant_id(tenant_id),
            collection=normalize_collection(collection),
            metadata=metadata,
            force_reindex=force_reindex,
            max_concurrency=max_concurrency,
            limits=archive_limits,
            event_sink=self._event_sink,
        )

    @operation(ingest=True)
    async def add_url(
        self,
        url: str,
        *,
        collection: str | None = None,
        tenant_id: str | None = None,
        document_id: str | None = None,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        fetch_client: FetchClient | None = None,
        fetch_policy: FetchSecurityPolicy | None = None,
        fetch_limits: FetchLimits | None = None,
    ) -> IngestedDocument:
        tenant_id, collection = validate_ingest_scope(
            tenant_id, collection, self._config.collection_policy
        )
        return await ingest_remote_url(
            url,
            ingest_bytes=self._add_bytes,
            tenant_id=normalize_tenant_id(tenant_id),
            collection=normalize_collection(collection),
            event_sink=self._event_sink,
            document_id=document_id,
            metadata=metadata,
            force_reindex=force_reindex,
            fetch_client=fetch_client,
            fetch_policy=fetch_policy,
            fetch_limits=fetch_limits,
            max_document_bytes=self._config.ingest.max_document_bytes,
        )

    async def _delete_document(
        self,
        *,
        document_id: str,
        collection: str | None = None,
        tenant_id: str | None = None,
    ) -> DeleteDocumentResult:
        return await self._ingest.delete_document(
            document_id=document_id,
            tenant_id=normalize_tenant_id(tenant_id),
            collection=normalize_collection(collection),
        )

    @operation()
    async def delete_collection(
        self,
        *,
        collection: str | None = None,
        tenant_id: str | None = None,
    ) -> None:
        await self._ingest.delete_collection(
            tenant_id=normalize_tenant_id(tenant_id),
            collection=normalize_collection(collection),
        )

    @operation()
    async def delete_tenant(self, *, tenant_id: str) -> None:
        await self._ingest.delete_tenant(tenant_id=tenant_id)

    @operation(search=True)
    async def search(
        self,
        *,
        query: str,
        collection: str | None = None,
        collections: Sequence[str] | None = None,
        tenant_id: str | None = None,
        limit: int = DEFAULT_SEARCH_LIMIT,
        content_types: list[str] | None = None,
        document_ids: list[str] | None = None,
        rerank: bool = DEFAULT_RERANK,
        mode: SearchMode | None = None,
        metadata_filter: Filter | None = None,
        rerank_budget: RerankBudget | None = None,
        audit_context: AuditContext | None = None,
    ) -> list[SearchResult]:
        from rag_core.search.runner import SearchRequest

        request = SearchRequest(
            query=query,
            tenant_id=normalize_tenant_id(tenant_id),
            collections=resolve_collections_argument(
                collection=collection, collections=collections, caller="Engine.search"
            ),
            limit=limit,
            content_types=content_types,
            document_ids=document_ids,
            rerank=rerank,
            metadata_filter=metadata_filter,
            rerank_budget=rerank_budget,
            mode=mode or ("hybrid" if self._config.hybrid else "dense"),
            audit_context=audit_context,
        )
        return await self._search.search(request)

    @operation(search=True)
    async def context(
        self,
        *,
        query: str,
        collection: str | None = None,
        collections: Sequence[str] | None = None,
        tenant_id: str | None = None,
        limit: int = DEFAULT_CONTEXT_LIMIT,
        content_types: list[str] | None = None,
        document_ids: list[str] | None = None,
        rerank: bool = DEFAULT_RERANK,
        mode: SearchMode | None = None,
        metadata_filter: Filter | None = None,
        rerank_budget: RerankBudget | None = None,
        max_chars: int | None = None,
        max_tokens: int | None = None,
        deduplicate: bool = False,
        audit_context: AuditContext | None = None,
    ) -> Context:
        from rag_core.search.runner import SearchRequest

        request = SearchRequest(
            query=query,
            tenant_id=normalize_tenant_id(tenant_id),
            collections=resolve_collections_argument(
                collection=collection, collections=collections, caller="Engine.context"
            ),
            limit=limit,
            content_types=content_types,
            document_ids=document_ids,
            rerank=rerank,
            metadata_filter=metadata_filter,
            rerank_budget=rerank_budget,
            mode=mode or ("hybrid" if self._config.hybrid else "dense"),
            audit_context=audit_context,
        )
        return await self._search.context(
            request, max_chars=max_chars, max_tokens=max_tokens, deduplicate=deduplicate
        )

    ensure_ready = operation()(_ensure_ready)
    prepare_bytes = operation(ingest=True)(_prepare_bytes)
    add_bytes = operation(ingest=True)(_add_bytes)
    add_file = operation(ingest=True)(_add_file)
    delete_document = operation()(_delete_document)


class _IngestCallbacks:
    """Private callbacks covered by an admitted batch's whole-operation lease."""

    def __init__(self, engine: Engine) -> None:
        self.ensure_ready = engine._ensure_ready
        self.add_file = engine._add_file
        self.add_bytes = engine._add_bytes
        self.delete_document = engine._delete_document


__all__ = ["Engine"]
