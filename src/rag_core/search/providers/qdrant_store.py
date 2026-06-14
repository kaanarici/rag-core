"""Qdrant vector store with hardened writes, adaptive batching, and health checks."""

from __future__ import annotations

import logging
from typing import Sequence

from rag_core.search.policy import DEFAULT_POLICY, VectorStorePolicy
from rag_core.search.provider_protocols import StoreCapabilities
from rag_core.search.query_plan import QueryPlan
from rag_core.search.request_models import (
    DeleteFilter,
    SearchQuery,
    StoredDocumentRecord,
)
from rag_core.search.vector_models import (
    SearchResult,
    VectorPoint,
)

from .qdrant_delete import delete_qdrant_filter, delete_qdrant_point_ids
from .qdrant_documents import get_qdrant_chunks_by_index, get_qdrant_document_record
from .qdrant_health import (
    _build_base_health as _build_base_health,
    _collection_query_plan_capabilities as _collection_query_plan_capabilities,
    _build_healthy_health as _build_healthy_health,
    _build_unhealthy_health as _build_unhealthy_health,
    _extract_optimizer_ok as _extract_optimizer_ok,
    check_qdrant_health,
)
from .qdrant_lifecycle import (
    QdrantCollectionState,
    ensure_qdrant_collection_ready,
)
from .qdrant_payloads import _score_result_value as _score_result_value
from .qdrant_query_plan import (
    validate_qdrant_query_plan_sparse_channels,
)
from .qdrant_search import (
    qdrant_default_query_plan_for_sparse_channels,
    search_qdrant_points,
)
from .qdrant_runtime import create_qdrant_adapter_runtime
from .qdrant_store_guards import (
    validate_qdrant_delete_filter,
    validate_qdrant_query_plan_preflight,
    validate_qdrant_search_request,
)
from .vector_store_capabilities import (
    QDRANT_VECTOR_STORE_CAPABILITY_SPEC,
    QDRANT_VECTOR_STORE_PROVIDER_SPEC,
    qdrant_query_plan_capabilities_for_sparse_names,
)
from .qdrant_write import upsert_qdrant_point_batches
from .registry import VECTOR_STORES
from .vector_dimensions import (
    validate_point_dense_dimensions,
)

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """Qdrant adapter implementing the full ``VectorStore`` capability surface."""

    def __init__(
        self,
        url: str | None,
        api_key: str | None,
        collection_name: str,
        location: str | None = None,
        dense_dimensions: int = 3072,
        quantization_enabled: bool = True,
        policy: VectorStorePolicy = DEFAULT_POLICY,
        embedding_model: str | None = None,
    ) -> None:
        runtime = create_qdrant_adapter_runtime(
            url=url,
            api_key=api_key,
            location=location,
            collection_name=collection_name,
            dense_dimensions=dense_dimensions,
            quantization_enabled=quantization_enabled,
            policy=policy,
            logger=logger,
            embedding_model=embedding_model,
        )
        self._client = runtime.client
        self._config = runtime.config
        self._policy = policy
        self._collection_state = QdrantCollectionState()
        self._write_sem = runtime.write_sem
        self._latency = runtime.latency

    async def __aenter__(self) -> "QdrantVectorStore":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    @property
    def capabilities(self) -> StoreCapabilities:
        query_plan_capabilities = QDRANT_VECTOR_STORE_CAPABILITY_SPEC.query_plan
        if self._collection_state.ready:
            query_plan_capabilities = qdrant_query_plan_capabilities_for_sparse_names(
                self._collection_state.available_sparse_vector_names,
            )
        return QDRANT_VECTOR_STORE_CAPABILITY_SPEC.to_store_capabilities(
            dense_vector_dimensions=self._config.dimensions,
            query_plan=query_plan_capabilities,
        )

    async def close(self) -> None:
        await self._client.close()

    async def ensure_collection(self) -> None:
        await ensure_qdrant_collection_ready(
            client=self._client,
            config=self._config,
            state=self._collection_state,
            logger=logger,
        )

    async def check_health(self) -> dict[str, object]:
        return await check_qdrant_health(
            client=self._client,
            collection_name=self._config.collection_name,
            dimensions=self._config.dimensions,
            latency=self._latency,
            logger=logger,
        )

    def validate_query_plan(self, plan: QueryPlan) -> None:
        validate_qdrant_query_plan_preflight(plan)

    def default_query_plan(self, *, result_limit: int) -> QueryPlan:
        return qdrant_default_query_plan_for_sparse_channels(
            result_limit=result_limit,
            sparse_channels=self._collection_state.available_sparse_vector_names,
        )

    async def prepare_query_plan(self, plan: QueryPlan) -> None:
        validate_qdrant_query_plan_preflight(plan)
        await self.ensure_collection()
        validate_qdrant_query_plan_sparse_channels(
            plan,
            available_sparse_names=self._collection_state.available_sparse_vector_names,
        )

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        if not points:
            return
        validate_point_dense_dimensions(
            points,
            dense_dimensions=self._config.dimensions,
            provider_name=QDRANT_VECTOR_STORE_PROVIDER_SPEC.name,
        )
        await self.ensure_collection()

        await upsert_qdrant_point_batches(
            client=self._client,
            collection_name=self._config.collection_name,
            dimensions=self._config.dimensions,
            latency=self._latency,
            max_batch_size=self._config.max_batch_size,
            write_sem=self._write_sem,
            points=points,
            available_sparse_vector_names=(
                self._collection_state.available_sparse_vector_names
            ),
        )

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        namespace = validate_qdrant_search_request(
            query,
            dense_dimensions=self._config.dimensions,
        )
        if query.has_empty_allowlist():
            return []
        await self.ensure_collection()

        return await search_qdrant_points(
            client=self._client,
            collection_name=self._config.collection_name,
            query=query,
            namespace=namespace,
            policy=self._policy,
            available_sparse_vector_names=(
                self._collection_state.available_sparse_vector_names
            ),
        )

    async def delete(self, filter: DeleteFilter) -> None:
        namespace = validate_qdrant_delete_filter(filter)
        await self.ensure_collection()

        await delete_qdrant_filter(
            client=self._client,
            collection_name=self._config.collection_name,
            filter_values=filter,
            namespace=namespace,
            policy=self._policy,
        )

    async def delete_point_ids(self, point_ids: Sequence[str]) -> None:
        if not point_ids:
            return
        await self.ensure_collection()
        await delete_qdrant_point_ids(
            client=self._client,
            collection_name=self._config.collection_name,
            point_ids=point_ids,
        )

    async def get_document_record(
        self,
        *,
        namespace: str,
        corpus_id: str,
        document_id: str | None = None,
        document_key: str | None = None,
    ) -> StoredDocumentRecord | None:
        await self.ensure_collection()
        return await get_qdrant_document_record(
            client=self._client,
            collection_name=self._config.collection_name,
            namespace=namespace,
            corpus_id=corpus_id,
            document_id=document_id,
            document_key=document_key,
            policy=self._policy,
        )

    async def get_chunks_by_index(
        self,
        *,
        namespace: str,
        corpus_id: str,
        document_id: str,
        chunk_indices: Sequence[int],
    ) -> list[SearchResult]:
        await self.ensure_collection()
        return await get_qdrant_chunks_by_index(
            client=self._client,
            collection_name=self._config.collection_name,
            namespace=namespace,
            corpus_id=corpus_id,
            document_id=document_id,
            chunk_indices=chunk_indices,
            policy=self._policy,
        )


VECTOR_STORES.register(
    QDRANT_VECTOR_STORE_PROVIDER_SPEC.name,
    lambda **kw: QdrantVectorStore(**kw),
)
