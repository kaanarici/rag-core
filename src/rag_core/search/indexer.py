"""DocumentIndexer: the indexing entry point and dense-vector validation."""

from __future__ import annotations

from dataclasses import replace

from rag_core.search.provider_protocols import EmbeddingProvider, VectorStore
import logging
from typing import TYPE_CHECKING
from rag_core.config.embedding_config import DEFAULT_EMBEDDING_BATCH_SIZE
from rag_core.events.emit import emit_event, now_ms
from rag_core.events.types import IndexUpserted
from rag_core.search.policy import CollectionPolicy, DEFAULT_POLICY, VectorStorePolicy
from .indexer_prepare import prepare_index_data, resolve_chunks
from .indexer_prepare import validate_embedding_batch_size
from .indexer_models import DeleteAck, IndexRequest, IndexResult
from .indexer_points import build_points
from rag_core.search.provider_protocols import (
    SparseEmbedder,
)
from rag_core.search.request_models import DeleteFilter, StoredDocumentRecord
from rag_core.search.policy import validate_ingest_scope


def validate_index_tenant_id(tenant_id: str) -> str:
    tenant_id_scoped = tenant_id.strip()
    if not tenant_id_scoped:
        raise ValueError("tenant_id is required for indexing")
    return tenant_id_scoped


def validate_delete_scope(tenant_id: str, collection: str) -> tuple[str, str]:
    tenant_id_scoped = tenant_id.strip()
    if not tenant_id_scoped:
        raise ValueError("tenant_id is required for delete_document")
    collection_scoped = collection.strip()
    if not collection_scoped:
        raise ValueError("collection is required for delete_document")
    return tenant_id_scoped, collection_scoped


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


def _store_dense_dimensions(vector_store: VectorStore) -> int | None:
    value = vector_store.capabilities.dense_vector_dimensions
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("vector store dense_vector_dimensions must be an integer")
    if value <= 0:
        raise ValueError("vector store dense_vector_dimensions must be positive")
    return value


if TYPE_CHECKING:
    from rag_core.events.sink import EventSink

logger = logging.getLogger(__name__)


class DocumentIndexer:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        sparse_embedder: SparseEmbedder | None,
        vector_store: VectorStore,
        event_sink: "EventSink | None" = None,
        policy: VectorStorePolicy = DEFAULT_POLICY,
        embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        *,
        collection_policy: CollectionPolicy | None = None,
    ) -> None:
        self._embedding = embedding_provider
        self._sparse = sparse_embedder
        self._store = vector_store
        self._store.bind_embedding_space(embedding_provider.model_name)
        self._event_sink = event_sink
        self._policy = policy
        self._collection_policy = collection_policy
        self._embedding_batch_size = validate_embedding_batch_size(embedding_batch_size)

    async def index_document(
        self,
        req: IndexRequest,
        *,
        event_sink: "EventSink | None" = None,
    ) -> IndexResult:
        # Per-call ``event_sink`` overrides the constructor-bound sink so a
        # caller-supplied audit-correlation wrapper (``CorrelatedSink``)
        # is propagated through the prepare → upsert → emit chain without
        # mutating the long-lived indexer instance.
        active_sink = event_sink if event_sink is not None else self._event_sink
        tenant_id, collection = validate_ingest_scope(
            req.tenant_id, req.collection, self._collection_policy
        )
        if not req.document_id or not req.document_id.strip():
            raise ValueError("document_id is required for indexing")
        req = replace(
            req,
            tenant_id=tenant_id,
            collection=collection,
            document_id=req.document_id.strip(),
        )
        upsert_started_ms = now_ms()
        chunks = resolve_chunks(req)
        validate_embedding_store_dimensions(self._embedding, self._store)
        prepared = await prepare_index_data(
            req=req,
            chunks=chunks,
            embedding_provider=self._embedding,
            sparse_embedder=self._sparse,
            event_sink=active_sink,
            embedding_batch_size=self._embedding_batch_size,
        )
        points = build_points(
            req=req,
            tenant_id=tenant_id,
            prepared=prepared,
            policy=self._policy,
        )
        await self._store.replace_document(
            StoredDocumentRecord(
                document_id=req.document_id,
                tenant_id=tenant_id,
                collection=collection,
                document_key=req.document_key,
                content_sha256=req.content_sha256,
                ingest_fingerprint=req.ingest_fingerprint,
                processing_version=req.processing_version,
                chunk_count=len(points),
            ),
            points,
        )

        emit_event(
            active_sink,
            IndexUpserted(
                tenant_id=tenant_id,
                collection=req.collection,
                document_id=req.document_id,
                point_count=len(points),
                duration_ms=now_ms() - upsert_started_ms,
            ),
        )
        logger.info("Indexed %d chunks", len(points))
        return IndexResult(
            document_id=req.document_id,
            chunk_count=len(points),
            document_key=req.document_key,
            content_sha256=req.content_sha256,
        )

    async def delete_document(
        self,
        document_id: str,
        tenant_id: str,
        *,
        collection: str,
    ) -> DeleteAck:
        """Delete all chunks for a document from the vector store.

        Per-document delete only. Refuses empty / blank ``document_id`` so a
        formatting bug cannot silently widen to ``DeleteFilter(collection=..)``
        and clear the whole collection (caught by ``DeleteFilter.__post_init__``
        too, but raised earlier here with a clearer message).

        Returns a ``DeleteAck`` so the engine can populate
        ``DeleteDocumentResult.vector_store_acked`` from the store's actual ack
        rather than the engine's optimism. Raises propagate.
        """
        tenant_id_scoped, collection_scoped = validate_delete_scope(
            tenant_id, collection
        )
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(
                "document_id is required for delete_document; pass a non-empty "
                "string. For collection-wide or tenant-wide deletes use the "
                "explicit delete_collection / delete_tenant facade helpers."
            )
        if self._collection_policy is not None:
            self._collection_policy.validate_delete(
                tenant_id=tenant_id_scoped,
                collection=collection_scoped,
            )
        await self._store.delete(
            DeleteFilter(
                tenant_id=tenant_id_scoped,
                collection=collection_scoped,
                document_id=document_id.strip(),
            ),
        )
        # Most adapters don't expose a deleted-point count without a follow-up
        # read; surface ``-1`` (unknown) and rely on absence-of-exception as the
        # success signal. The engine fields ``succeeded`` straight into
        # ``DeleteDocumentResult.vector_store_acked`` so callers see the store ack.
        return DeleteAck(succeeded=True, deleted_point_count=-1)

    async def delete_collection(
        self,
        *,
        tenant_id: str,
        collection: str,
    ) -> DeleteAck:
        """Explicit collection-wide delete.

        Callers must reach this method deliberately; constructing a
        ``DeleteFilter`` with no ``document_id`` (the former silent path) is
        forbidden at the engine seam by ``DeleteFilter.__post_init__``.
        """
        tenant_id_scoped, collection_scoped = validate_delete_scope(
            tenant_id, collection
        )
        if self._collection_policy is not None:
            self._collection_policy.validate_delete(
                tenant_id=tenant_id_scoped,
                collection=collection_scoped,
            )
        await self._store.delete(
            DeleteFilter(
                tenant_id=tenant_id_scoped,
                collection=collection_scoped,
                document_id=None,
            ),
        )
        return DeleteAck(succeeded=True, deleted_point_count=-1)


__all__ = [
    "DeleteAck",
    "DocumentIndexer",
    "IndexRequest",
    "IndexResult",
]
