"""In-memory vector store for tests and lightweight local runs.

Implements the ``VectorStore`` surface in pure Python for dense/sparse hybrid
retrieval with RRF fusion over an in-memory point map. Useful for protocol
validation, not intended as a production backend.
"""

from __future__ import annotations

from typing import Sequence

from rag_core.search.policy import DEFAULT_POLICY, VectorStorePolicy
from rag_core.search.query_plan import DenseChannel
from rag_core.search.query_plan import Prefetch
from rag_core.search.query_plan import QueryPlan
from rag_core.search.stored_payload import payload_to_result
from rag_core.search.types import (
    DeleteFilter,
    MetadataFilterCapabilities,
    SearchQuery,
    SearchResult,
    StoreCapabilities,
    StoredDocumentRecord,
    VectorPoint,
)

from .memory_documents import get_memory_document_record
from .memory_filters import matches_memory_delete_filter, matches_memory_search_filter
from .memory_query_plan import rank_memory_points
from .memory_query_plan import validate_memory_query_plan
from .memory_query_scoring import MemoryPoint
from .query_plan_capabilities import MEMORY_QUERY_PLAN_CAPABILITIES
from .registry import VECTOR_STORES
from .vector_dimensions import (
    validate_point_dense_dimensions,
    validate_query_dense_dimensions,
)


class InMemoryVectorStore:
    """Pure-Python ``VectorStore`` adapter for tests and lightweight runs."""

    def __init__(self, policy: VectorStorePolicy = DEFAULT_POLICY) -> None:
        self._points: dict[str, MemoryPoint] = {}
        self._policy = policy
        self._dense_dimensions: int | None = None

    @property
    def capabilities(self) -> StoreCapabilities:
        return StoreCapabilities(
            per_point_delete=True,
            document_record_lookup=True,
            dense_vector_dimensions=self._dense_dimensions,
            query_plan=MEMORY_QUERY_PLAN_CAPABILITIES,
            metadata_filter=MetadataFilterCapabilities(
                term=True,
                in_=True,
                numeric_range=True,
                string_range=True,
                geo=True,
                boolean=True,
            ),
        )

    async def __aenter__(self) -> "InMemoryVectorStore":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        self._points.clear()
        self._dense_dimensions = None

    async def ensure_collection(self) -> None:
        return None

    async def check_health(self) -> dict[str, object]:
        return {
            "healthy": True,
            "backend": "memory",
            "points_count": len(self._points),
        }

    def validate_query_plan(self, plan: QueryPlan) -> None:
        validate_memory_query_plan(plan)

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        self._validate_point_dimensions(points)
        for point in points:
            self._points[point.id] = MemoryPoint(
                id=point.id,
                dense=list(point.dense_vector),
                sparse=dict(point.all_sparse_vectors()),
                payload=dict(point.payload),
            )

    async def delete(self, filter: DeleteFilter) -> None:
        namespace = (filter.namespace or "").strip()
        if not namespace:
            raise ValueError("namespace is required for delete")

        keep: dict[str, MemoryPoint] = {}
        for stored in self._points.values():
            if not matches_memory_delete_filter(
                stored,
                filter,
                namespace=namespace,
                policy=self._policy,
            ):
                keep[stored.id] = stored
        self._points = keep

    async def delete_point_ids(self, point_ids: Sequence[str]) -> None:
        for point_id in point_ids:
            self._points.pop(point_id, None)

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        namespace = query.namespace.strip()
        if not namespace:
            raise ValueError("namespace is required for search")
        if query.query_plan is not None and _query_plan_uses_dense(query.query_plan):
            if not query.dense_vector:
                raise ValueError(
                    "InMemoryVectorStore dense query vector is required for dense query plans"
                )
        if self._dense_dimensions is not None and _query_uses_dense_vector(query):
            validate_query_dense_dimensions(
                query.dense_vector,
                dense_dimensions=self._dense_dimensions,
                backend="InMemoryVectorStore",
            )

        candidates = [
            stored
            for stored in self._points.values()
            if matches_memory_search_filter(
                stored,
                query,
                namespace=namespace,
                policy=self._policy,
            )
        ]

        results: list[SearchResult] = []
        for point_id, score in rank_memory_points(query, candidates):
            stored = self._points.get(point_id)
            if stored is None:
                continue
            results.append(
                payload_to_result(
                    point_id=stored.id,
                    payload=stored.payload,
                    score=score,
                    policy=self._policy,
                )
            )
        return results

    async def get_document_record(
        self,
        *,
        namespace: str,
        corpus_id: str,
        document_id: str | None = None,
        document_key: str | None = None,
    ) -> StoredDocumentRecord | None:
        return get_memory_document_record(
            self._points.values(),
            namespace=namespace,
            corpus_id=corpus_id,
            document_id=document_id,
            document_key=document_key,
            policy=self._policy,
        )

    def _validate_point_dimensions(self, points: Sequence[VectorPoint]) -> None:
        incoming_dimensions = {
            len(point.dense_vector) for point in points if point.dense_vector
        }
        if not incoming_dimensions:
            return
        if len(incoming_dimensions) > 1:
            expected = min(incoming_dimensions)
            validate_point_dense_dimensions(
                points,
                dense_dimensions=expected,
                backend="InMemoryVectorStore",
            )
        dimensions = next(iter(incoming_dimensions))
        if self._dense_dimensions is None:
            self._dense_dimensions = dimensions
            return
        validate_point_dense_dimensions(
            points,
            dense_dimensions=self._dense_dimensions,
            backend="InMemoryVectorStore",
        )


def _query_uses_dense_vector(query: SearchQuery) -> bool:
    if query.query_plan is None:
        return bool(query.dense_vector)
    return _query_plan_uses_dense(query.query_plan)


def _query_plan_uses_dense(plan: QueryPlan) -> bool:
    return any(_prefetch_uses_dense_vector(prefetch) for prefetch in plan.prefetches)


def _prefetch_uses_dense_vector(prefetch: Prefetch) -> bool:
    if isinstance(prefetch.channel, DenseChannel):
        return True
    return any(_prefetch_uses_dense_vector(nested) for nested in prefetch.nested)


VECTOR_STORES.register("memory", lambda **kw: InMemoryVectorStore(**kw))
