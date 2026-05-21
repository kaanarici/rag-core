from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rag_core.config.embedding_config import DEFAULT_EMBEDDING_BATCH_SIZE
from rag_core.events.emit import emit_event, now_ms
from rag_core.events.types import IndexUpserted
from rag_core.search.policy import DEFAULT_POLICY, VectorStorePolicy

from .indexer_embeddings import prepare_index_data
from .indexer_models import IndexRequest, IndexResult
from .indexer_points import build_points, build_stale_point_ids
from .indexer_validation import (
    validate_delete_scope,
    validate_embedding_batch_size,
    validate_embedding_store_dimensions,
    validate_index_namespace,
)
from rag_core.search.types import (
    DeleteFilter,
    EmbeddingProvider,
    SparseEmbedder,
    VectorStore,
)

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink

logger = logging.getLogger(__name__)


class DocumentIndexer:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        sparse_embedder: SparseEmbedder,
        vector_store: VectorStore,
        event_sink: "EventSink | None" = None,
        policy: VectorStorePolicy = DEFAULT_POLICY,
        embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    ) -> None:
        self._embedding = embedding_provider
        self._sparse = sparse_embedder
        self._store = vector_store
        self._event_sink = event_sink
        self._policy = policy
        self._embedding_batch_size = validate_embedding_batch_size(
            embedding_batch_size
        )

    async def index_document(self, req: IndexRequest) -> IndexResult:
        namespace = validate_index_namespace(req.namespace)
        upsert_started_ms = now_ms()
        prepared = await prepare_index_data(
            req=req,
            embedding_provider=self._embedding,
            sparse_embedder=self._sparse,
            event_sink=self._event_sink,
            embedding_batch_size=self._embedding_batch_size,
        )
        if not prepared.chunks:
            if req.document_id:
                await self._store.delete(
                    DeleteFilter(
                        namespace=namespace,
                        corpus_id=req.corpus_id,
                        document_id=req.document_id,
                    )
                )
            return IndexResult(
                document_id=req.document_id,
                chunk_count=0,
                point_ids=[],
                point_payloads=[],
                document_key=req.document_key,
                content_sha256=req.content_sha256,
            )

        validate_embedding_store_dimensions(self._embedding, self._store)
        points, point_ids = build_points(
            req=req,
            namespace=namespace,
            prepared=prepared,
            policy=self._policy,
        )
        stale_point_ids = build_stale_point_ids(
            req, new_chunk_count=len(points), policy=self._policy
        )
        store = self._store
        per_point_delete = store.capabilities.per_point_delete
        if stale_point_ids and not per_point_delete:
            raise RuntimeError(
                "Vector store cannot safely replace a document with fewer chunks "
                "because it does not support per-point deletes"
            )
        if stale_point_ids and per_point_delete:
            await store.delete_point_ids(stale_point_ids)
        await store.upsert(points)

        emit_event(
            self._event_sink,
            IndexUpserted(
                namespace=namespace,
                corpus_id=req.corpus_id,
                document_id=req.document_id,
                point_count=len(points),
                duration_ms=now_ms() - upsert_started_ms,
            ),
        )
        logger.info("Indexed %d chunks", len(points))
        return IndexResult(
            document_id=req.document_id,
            chunk_count=len(points),
            point_ids=point_ids,
            point_payloads=[dict(point.payload) for point in points],
            document_key=req.document_key,
            content_sha256=req.content_sha256,
        )

    async def delete_document(
        self,
        document_id: str,
        namespace: str,
        *,
        corpus_id: str,
    ) -> None:
        """Delete all chunks for a document from the vector store."""
        namespace_scoped, corpus_scoped = validate_delete_scope(namespace, corpus_id)
        await self._store.delete(
            DeleteFilter(
                namespace=namespace_scoped,
                corpus_id=corpus_scoped,
                document_id=document_id,
            ),
        )


__all__ = ["DocumentIndexer", "IndexRequest", "IndexResult"]
