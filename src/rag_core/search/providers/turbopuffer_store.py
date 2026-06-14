"""TurboPuffer vector store adapter."""

from __future__ import annotations

from collections.abc import Sequence

from rag_core.config.vector_store_config import (
    DEFAULT_TURBOPUFFER_DELETE_CONTINUATION_LIMIT,
    DEFAULT_TURBOPUFFER_DISTANCE_METRIC,
)
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

from .vector_store_capabilities import (
    TURBOPUFFER_VECTOR_STORE_CAPABILITY_SPEC,
    TURBOPUFFER_VECTOR_STORE_PROVIDER_SPEC,
)
from .registry import VECTOR_STORES
from .turbopuffer_client import (
    TurboPufferNamespace,
    close_turbopuffer_client,
    resolve_turbopuffer_namespace,
)
from .turbopuffer_config import (
    DEFAULT_TURBOPUFFER_WRITE_BATCH_SIZE,
    build_turbopuffer_config,
    owns_turbopuffer_client,
)
from .turbopuffer_health import _build_healthy_health, _build_unhealthy_health
from .turbopuffer_documents import get_turbopuffer_document_record
from .turbopuffer_query_plan import _supported_query_plan_limit
from .turbopuffer_search import search_turbopuffer_points
from .turbopuffer_write import (
    delete_turbopuffer_filter,
    delete_turbopuffer_point_ids,
    upsert_turbopuffer_points,
)


class TurboPufferVectorStore:
    """TurboPuffer adapter for the baseline ``VectorStore`` contract."""

    def __init__(
        self,
        *,
        namespace: str,
        dense_dimensions: int,
        api_key: str | None = None,
        region: str | None = None,
        base_url: str | None = None,
        distance_metric: str = DEFAULT_TURBOPUFFER_DISTANCE_METRIC,
        write_batch_size: int = DEFAULT_TURBOPUFFER_WRITE_BATCH_SIZE,
        delete_continuation_limit: int = DEFAULT_TURBOPUFFER_DELETE_CONTINUATION_LIMIT,
        client: object | None = None,
        namespace_client: object | None = None,
        policy: VectorStorePolicy = DEFAULT_POLICY,
    ) -> None:
        self._api_key = api_key
        self._config = build_turbopuffer_config(
            namespace=namespace,
            dense_dimensions=dense_dimensions,
            region=region,
            base_url=base_url,
            distance_metric=distance_metric,
            write_batch_size=write_batch_size,
            delete_continuation_limit=delete_continuation_limit,
            policy=policy,
        )
        self._policy = policy
        self._client = client
        self._namespace_client = namespace_client
        self._owns_client = owns_turbopuffer_client(
            client=client,
            namespace_client=namespace_client,
        )

    async def __aenter__(self) -> "TurboPufferVectorStore":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    @property
    def capabilities(self) -> StoreCapabilities:
        return TURBOPUFFER_VECTOR_STORE_CAPABILITY_SPEC.to_store_capabilities(
            dense_vector_dimensions=self._config.dense_dimensions,
        )

    def validate_query_plan(self, plan: QueryPlan) -> None:
        _supported_query_plan_limit(plan, fallback=1)

    def query_plan_needs_sparse_vectors(self, plan: QueryPlan | None) -> bool:
        return False

    async def ensure_collection(self) -> None:
        return None

    async def close(self) -> None:
        await close_turbopuffer_client(
            owns_client=self._owns_client,
            client=self._client,
        )

    async def check_health(self) -> dict[str, object]:
        try:
            metadata = await self._namespace().metadata()
        except Exception as exc:
            return _build_unhealthy_health(
                namespace=self._config.namespace,
                exc=exc,
            )
        return _build_healthy_health(
            namespace=self._config.namespace,
            metadata=metadata,
        )

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        await upsert_turbopuffer_points(
            namespace_client=self._namespace(),
            points=points,
            dense_dimensions=self._config.dense_dimensions,
            distance_metric=self._config.distance_metric,
            write_batch_size=self._config.write_batch_size,
            policy=self._policy,
        )

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        return await search_turbopuffer_points(
            namespace_client=self._namespace(),
            query=query,
            dense_dimensions=self._config.dense_dimensions,
            distance_metric=self._config.distance_metric,
            policy=self._policy,
        )

    async def delete(self, filter: DeleteFilter) -> None:
        await delete_turbopuffer_filter(
            namespace_client=self._namespace(),
            filter_values=filter,
            continuation_limit=self._config.delete_continuation_limit,
            policy=self._policy,
        )

    async def delete_point_ids(self, point_ids: Sequence[str]) -> None:
        await delete_turbopuffer_point_ids(
            namespace_client=self._namespace(),
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
        return await get_turbopuffer_document_record(
            namespace_client=self._namespace(),
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
        raise NotImplementedError(
            "TurboPufferVectorStore does not support chunk-index lookup"
        )

    def _namespace(self) -> TurboPufferNamespace:
        state = resolve_turbopuffer_namespace(
            namespace_client=self._namespace_client,
            client=self._client,
            namespace=self._config.namespace,
            api_key=self._api_key,
            region=self._config.region,
            base_url=self._config.base_url,
        )
        self._client = state.client
        self._namespace_client = state.namespace
        return state.namespace


VECTOR_STORES.register(
    TURBOPUFFER_VECTOR_STORE_PROVIDER_SPEC.name,
    lambda **kw: TurboPufferVectorStore(**kw),
)
