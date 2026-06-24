"""TurboPuffer vector store adapter."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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

from .registry import VECTOR_STORES
from .turbopuffer_client import (
    DEFAULT_TURBOPUFFER_WRITE_BATCH_SIZE,
    TurboPufferNamespace,
    _build_healthy_health,
    _build_unhealthy_health,
    build_turbopuffer_config,
    close_turbopuffer_client,
    owns_turbopuffer_client,
    resolve_turbopuffer_namespace,
    validate_turbopuffer_delete_continuation_limit,
    validate_turbopuffer_write_batch_size,
)
from .turbopuffer_payloads import (
    _delete_filter,
    _point_to_row,
    _schema,
    _validate_point_id,
)
from .turbopuffer_query import (
    _supported_query_plan_limit,
    get_turbopuffer_document_record,
    search_turbopuffer_points,
)
from .vector_dimensions import validate_point_dense_dimensions
from .vector_store_capabilities import (
    TURBOPUFFER_VECTOR_STORE_CAPABILITY_SPEC,
    TURBOPUFFER_VECTOR_STORE_PROVIDER_SPEC,
)


@dataclass(frozen=True)
class TurboPufferDeleteByFilterOutcome:
    writes_attempted: int
    exhausted: bool
    rows_remaining: bool


class TurboPufferDeleteByFilterExhausted(ValueError):
    def __init__(self, *, outcome: TurboPufferDeleteByFilterOutcome) -> None:
        self.outcome = outcome
        super().__init__(
            "turbopuffer delete by filter exhausted continuation limit "
            f"(writes_attempted={outcome.writes_attempted}, rows_remaining={outcome.rows_remaining})"
        )


async def upsert_turbopuffer_points(
    *,
    namespace_client: TurboPufferNamespace,
    points: Sequence[VectorPoint],
    dense_dimensions: int,
    distance_metric: str,
    write_batch_size: int,
    policy: VectorStorePolicy,
) -> None:
    if not points:
        return
    write_batch_size = validate_turbopuffer_write_batch_size(write_batch_size)
    validate_point_dense_dimensions(
        points,
        dense_dimensions=dense_dimensions,
        provider_name=TURBOPUFFER_VECTOR_STORE_PROVIDER_SPEC.name,
    )
    rows = [_point_to_row(point, policy=policy) for point in points]
    schema = _schema(dense_dimensions, policy=policy)
    for index in range(0, len(rows), write_batch_size):
        await namespace_client.write(
            upsert_rows=rows[index : index + write_batch_size],
            schema=schema,
            distance_metric=distance_metric,
        )


async def delete_turbopuffer_filter(
    *,
    namespace_client: TurboPufferNamespace,
    filter_values: DeleteFilter,
    policy: VectorStorePolicy,
    continuation_limit: int = DEFAULT_TURBOPUFFER_DELETE_CONTINUATION_LIMIT,
    raise_on_exhausted: bool = True,
) -> TurboPufferDeleteByFilterOutcome:
    namespace = (filter_values.namespace or "").strip()
    if not namespace:
        raise ValueError("namespace is required for delete")
    continuation_limit = validate_turbopuffer_delete_continuation_limit(
        continuation_limit
    )
    delete_filter = _delete_filter(
        filter_values=filter_values,
        namespace=namespace,
        policy=policy,
    )
    for writes_attempted in range(1, continuation_limit + 1):
        response = await namespace_client.write(
            delete_by_filter=delete_filter,
            delete_by_filter_allow_partial=True,
        )
        rows_remaining = getattr(response, "rows_remaining", False)
        if not isinstance(rows_remaining, bool):
            raise ValueError(
                "turbopuffer delete response returned invalid rows_remaining"
            )
        if not rows_remaining:
            return TurboPufferDeleteByFilterOutcome(
                writes_attempted=writes_attempted,
                exhausted=False,
                rows_remaining=False,
            )
    exhausted = TurboPufferDeleteByFilterOutcome(
        writes_attempted=continuation_limit,
        exhausted=True,
        rows_remaining=True,
    )
    if raise_on_exhausted:
        raise TurboPufferDeleteByFilterExhausted(outcome=exhausted)
    return exhausted


async def delete_turbopuffer_point_ids(
    *,
    namespace_client: TurboPufferNamespace,
    point_ids: Sequence[str],
) -> None:
    if not point_ids:
        return
    await namespace_client.write(
        deletes=[_validate_point_id(point_id) for point_id in point_ids]
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
        collection: str,
        document_id: str | None = None,
        document_key: str | None = None,
    ) -> StoredDocumentRecord | None:
        return await get_turbopuffer_document_record(
            namespace_client=self._namespace(),
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
