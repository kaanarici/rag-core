"""Filter construction helpers for the Qdrant vector store.

Field names come from the ``VectorStorePolicy`` so adapters can override
payload conventions without rewriting filter logic.
"""

from __future__ import annotations

from typing import Any, NoReturn, Sequence, cast

from qdrant_client import models as rest

from rag_core.search.filter_visitor import FilterTranslator
from rag_core.search.filters import (
    Filter,
    Geo,
    In,
    Range,
    Term,
)
from rag_core.search.policy import DEFAULT_POLICY, VectorStorePolicy
from rag_core.search.query_plan import UnsupportedQueryStage
from rag_core.search.request_models import (
    DeleteFilter,
    SearchQuery,
)


def metadata_filter_to_qdrant(filter: Filter) -> rest.Filter:
    """Translate the engine's ``Filter`` AST into a Qdrant ``rest.Filter``."""

    return rest.Filter(must=cast(Any, [_QdrantFilterTranslator().translate(filter)]))


class _QdrantFilterTranslator(FilterTranslator[object]):
    def term(self, node: Term) -> object:
        return rest.FieldCondition(
            key=node.field,
            match=rest.MatchValue(value=cast(Any, node.value)),
        )

    def in_(self, node: In) -> object:
        return rest.FieldCondition(
            key=node.field,
            match=rest.MatchAny(any=cast(Any, list(node.values))),
        )

    def range_(self, node: Range) -> object:
        return rest.FieldCondition(
            key=node.field,
            range=rest.Range(
                gte=_qdrant_range_bound(node.gte),
                lte=_qdrant_range_bound(node.lte),
                gt=_qdrant_range_bound(node.gt),
                lt=_qdrant_range_bound(node.lt),
            ),
        )

    def geo(self, node: Geo) -> object:
        return rest.FieldCondition(
            key=node.field,
            geo_radius=rest.GeoRadius(
                center=rest.GeoPoint(lat=node.lat, lon=node.lon),
                radius=node.radius_m,
            ),
        )

    def and_(self, children: list[object]) -> object:
        return rest.Filter(must=cast(Any, children))

    def or_(self, children: list[object]) -> object:
        return rest.Filter(should=cast(Any, children))

    def not_(self, child: object) -> object:
        return rest.Filter(must_not=cast(Any, [child]))

    def unsupported(self, node: Filter) -> NoReturn:
        raise UnsupportedQueryStage(
            f"qdrant adapter cannot translate Filter node {type(node).__name__}"
        )


def _qdrant_range_bound(bound: float | str | None) -> float | None:
    if bound is None:
        return None
    if isinstance(bound, str):
        raise UnsupportedQueryStage(
            "qdrant adapter cannot translate string Range filters"
        )
    return bound


def build_filter(must_conditions: Sequence[object]) -> rest.Filter:
    return rest.Filter(must=cast(Any, list(must_conditions)))


def build_search_filter(
    *,
    query: SearchQuery,
    namespace: str,
    policy: VectorStorePolicy = DEFAULT_POLICY,
) -> rest.Filter:
    must_conditions: list[object] = [
        rest.FieldCondition(
            key=policy.namespace_field,
            match=rest.MatchValue(value=namespace),
        ),
    ]

    if query.collections:
        must_conditions.append(
            rest.FieldCondition(
                key=policy.collection_field,
                match=rest.MatchAny(any=query.collections),
            )
        )
    if query.content_types:
        must_conditions.append(
            rest.FieldCondition(
                key=policy.content_type_field,
                match=rest.MatchAny(any=query.content_types),
            )
        )
    if query.document_ids:
        must_conditions.append(
            rest.FieldCondition(
                key=policy.document_id_field,
                match=rest.MatchAny(any=query.document_ids),
            )
        )
    if query.metadata_filter is not None:
        must_conditions.append(metadata_filter_to_qdrant(query.metadata_filter))

    return build_filter(must_conditions)


def build_delete_filter(
    *,
    filter_values: DeleteFilter,
    namespace: str,
    policy: VectorStorePolicy = DEFAULT_POLICY,
) -> rest.Filter:
    must_conditions: list[object] = [
        rest.FieldCondition(
            key=policy.namespace_field,
            match=rest.MatchValue(value=namespace),
        ),
    ]

    if filter_values.collection:
        must_conditions.append(
            rest.FieldCondition(
                key=policy.collection_field,
                match=rest.MatchValue(value=filter_values.collection),
            )
        )
    if filter_values.document_id:
        must_conditions.append(
            rest.FieldCondition(
                key=policy.document_id_field,
                match=rest.MatchValue(value=filter_values.document_id),
            )
        )

    return build_filter(must_conditions)


def build_document_lookup_filter(
    *,
    namespace: str,
    collection: str,
    document_id: str | None,
    document_key: str | None,
    policy: VectorStorePolicy = DEFAULT_POLICY,
) -> rest.Filter:
    must_conditions: list[object] = [
        rest.FieldCondition(
            key=policy.namespace_field,
            match=rest.MatchValue(value=namespace),
        ),
        rest.FieldCondition(
            key=policy.collection_field,
            match=rest.MatchValue(value=collection),
        ),
    ]

    if document_id is not None:
        must_conditions.append(
            rest.FieldCondition(
                key=policy.document_id_field,
                match=rest.MatchValue(value=document_id),
            )
        )
    if document_key is not None:
        must_conditions.append(
            rest.FieldCondition(
                key=policy.document_key_field,
                match=rest.MatchValue(value=document_key),
            )
        )
    return build_filter(must_conditions)


def build_document_count_filter(
    *,
    namespace: str,
    collection: str,
    document_id: str,
    policy: VectorStorePolicy = DEFAULT_POLICY,
) -> rest.Filter:
    return build_filter(
        [
            rest.FieldCondition(
                key=policy.namespace_field,
                match=rest.MatchValue(value=namespace),
            ),
            rest.FieldCondition(
                key=policy.collection_field,
                match=rest.MatchValue(value=collection),
            ),
            rest.FieldCondition(
                key=policy.document_id_field,
                match=rest.MatchValue(value=document_id),
            ),
        ]
    )


def build_chunk_index_lookup_filter(
    *,
    namespace: str,
    collection: str,
    document_id: str,
    chunk_indices: Sequence[int],
    policy: VectorStorePolicy = DEFAULT_POLICY,
) -> rest.Filter:
    return build_filter(
        [
            rest.FieldCondition(
                key=policy.namespace_field,
                match=rest.MatchValue(value=namespace),
            ),
            rest.FieldCondition(
                key=policy.collection_field,
                match=rest.MatchValue(value=collection),
            ),
            rest.FieldCondition(
                key=policy.document_id_field,
                match=rest.MatchValue(value=document_id),
            ),
            rest.FieldCondition(
                key=policy.chunk_index_field,
                match=rest.MatchAny(any=list(chunk_indices)),
            ),
        ]
    )
