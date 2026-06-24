"""Qdrant vector store with hardened writes, adaptive batching, and health checks.

This module owns the public store entry plus the small construction helpers
(client creation and adapter runtime assembly) that only the store needs. The
shared qdrant constants and ``WriteLatencyTracker`` / ``compute_write_params``
live in ``qdrant_payloads`` (the lowest qdrant module) to keep those helpers
importable by the lower-level modules without an import cycle through this entry.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Sequence

from qdrant_client import AsyncQdrantClient

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

from .qdrant_collection import (
    QdrantAdapterConfig,
    QdrantCollectionState,
    _build_base_health as _build_base_health,
    _build_healthy_health as _build_healthy_health,
    _build_unhealthy_health as _build_unhealthy_health,
    _collection_fingerprint,
    _collection_query_plan_capabilities as _collection_query_plan_capabilities,
    _extract_optimizer_ok as _extract_optimizer_ok,
    check_qdrant_health,
    ensure_qdrant_collection_ready,
)
from .qdrant_payloads import (
    WriteLatencyTracker,
    _score_result_value as _score_result_value,
    compute_write_params,
    get_qdrant_chunks_by_index,
    get_qdrant_document_record,
)
from .qdrant_query import (
    qdrant_default_query_plan_for_sparse_channels,
    search_qdrant_points,
    validate_qdrant_delete_filter,
    validate_qdrant_query_plan_preflight,
    validate_qdrant_query_plan_sparse_channels,
    validate_qdrant_search_request,
)
from .qdrant_write import (
    delete_qdrant_filter,
    delete_qdrant_point_ids,
    upsert_qdrant_point_batches,
)
from .registry import VECTOR_STORES
from .vector_dimensions import (
    validate_point_dense_dimensions,
)
from .vector_store_capabilities import (
    QDRANT_VECTOR_STORE_CAPABILITY_SPEC,
    QDRANT_VECTOR_STORE_PROVIDER_SPEC,
    qdrant_query_plan_capabilities_for_sparse_names,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QdrantClientState:
    client: AsyncQdrantClient
    is_local: bool


def create_qdrant_client(
    *,
    url: str | None,
    api_key: str | None,
    location: str | None,
    timeout: int = 120,
) -> QdrantClientState:
    api_key = _normalize_api_key(api_key)
    if bool(url) == bool(location):
        raise ValueError(
            "QdrantVectorStore requires exactly one of url or location; "
            "use QdrantConfig(location=':memory:'), pass url=..., "
            "or inject vector_store=... into Engine."
        )

    if location is not None:
        if location != ":memory:":
            return QdrantClientState(
                client=AsyncQdrantClient(
                    path=location,
                    timeout=timeout,
                    check_compatibility=False,
                ),
                is_local=True,
            )
        return QdrantClientState(
            client=AsyncQdrantClient(
                location=location,
                timeout=timeout,
                check_compatibility=False,
            ),
            is_local=True,
        )

    if api_key is not None:
        return QdrantClientState(
            client=AsyncQdrantClient(url=url, api_key=api_key, timeout=timeout),
            is_local=False,
        )

    return QdrantClientState(
        client=AsyncQdrantClient(url=url, timeout=timeout),
        is_local=False,
    )


def _normalize_api_key(api_key: str | None) -> str | None:
    if api_key is None:
        return None
    stripped = api_key.strip()
    return stripped or None


@dataclass(frozen=True)
class QdrantAdapterRuntime:
    client: AsyncQdrantClient
    is_local: bool
    config: QdrantAdapterConfig
    write_sem: asyncio.Semaphore
    latency: WriteLatencyTracker


def create_qdrant_adapter_runtime(
    *,
    url: str | None,
    api_key: str | None,
    location: str | None,
    collection_name: str,
    dense_dimensions: int,
    quantization_enabled: bool,
    policy: VectorStorePolicy,
    logger: logging.Logger,
    embedding_model: str | None = None,
) -> QdrantAdapterRuntime:
    client_state = create_qdrant_client(
        url=url,
        api_key=api_key,
        location=location,
    )
    max_concurrent, max_batch_size = compute_write_params(dense_dimensions)
    config = QdrantAdapterConfig(
        collection_name=collection_name,
        dimensions=dense_dimensions,
        quantization_enabled=quantization_enabled,
        is_local=client_state.is_local,
        max_concurrent=max_concurrent,
        max_batch_size=max_batch_size,
        policy=policy,
        embedding_model=embedding_model,
    )
    _log_qdrant_adapter_initialized(logger, config=config)
    return QdrantAdapterRuntime(
        client=client_state.client,
        is_local=client_state.is_local,
        config=config,
        write_sem=asyncio.Semaphore(max_concurrent),
        latency=WriteLatencyTracker(),
    )


def _log_qdrant_adapter_initialized(
    logger: logging.Logger,
    *,
    config: QdrantAdapterConfig,
) -> None:
    logger.info(
        "QdrantVectorStore initialized: provider=%s dense_dimensions=%d "
        "max_concurrent=%d max_batch_size=%d quantization=%s local=%s "
        "collection_fingerprint=%s",
        QDRANT_VECTOR_STORE_PROVIDER_SPEC.name,
        config.dimensions,
        config.max_concurrent,
        config.max_batch_size,
        config.quantization_enabled,
        config.is_local,
        _collection_fingerprint(config.collection_name),
    )


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
        collection: str,
        document_id: str | None = None,
        document_key: str | None = None,
    ) -> StoredDocumentRecord | None:
        await self.ensure_collection()
        return await get_qdrant_document_record(
            client=self._client,
            collection_name=self._config.collection_name,
            namespace=namespace,
            collection=collection,
            document_id=document_id,
            document_key=document_key,
            policy=self._policy,
        )

    async def get_chunks_by_index(
        self,
        *,
        namespace: str,
        collection: str,
        document_id: str,
        chunk_indices: Sequence[int],
    ) -> list[SearchResult]:
        await self.ensure_collection()
        return await get_qdrant_chunks_by_index(
            client=self._client,
            collection_name=self._config.collection_name,
            namespace=namespace,
            collection=collection,
            document_id=document_id,
            chunk_indices=chunk_indices,
            policy=self._policy,
        )


VECTOR_STORES.register(
    QDRANT_VECTOR_STORE_PROVIDER_SPEC.name,
    lambda **kw: QdrantVectorStore(**kw),
)
