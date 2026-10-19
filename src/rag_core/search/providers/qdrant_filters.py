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
from rag_core.search.filters import UnsupportedFilter
from rag_core.search.request_models import (
    DeleteFilter,
    SearchQuery,
)


def metadata_filter_to_qdrant(filter: Filter) -> rest.Filter:
    """Translate the engine's ``Filter`` AST into a Qdrant ``rest.Filter``."""

    return rest.Filter(must=cast(Any, [_QdrantFilterTranslator().translate(filter)]))


class _QdrantFilterTranslator(FilterTranslator[object]):
    def term(self, node: Term) -> object:
        if isinstance(node.value, float):
            return self.range_(Range(node.field, gte=node.value, lte=node.value))
        return rest.FieldCondition(
            key=node.field,
            match=rest.MatchValue(value=cast(Any, node.value)),
        )

    def in_(self, node: In) -> object:
        if not (
            all(type(value) is str for value in node.values)
            or all(type(value) is int for value in node.values)
        ):
            return self.or_([self.term(Term(node.field, value)) for value in node.values])
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
        raise UnsupportedFilter(
            f"qdrant adapter cannot translate Filter node {type(node).__name__}"
        )


def _qdrant_range_bound(bound: float | str | None) -> float | None:
    if bound is None:
        return None
    if isinstance(bound, str):
        raise UnsupportedFilter("qdrant adapter cannot translate string Range filters")
    return bound


def build_filter(must_conditions: Sequence[object]) -> rest.Filter:
    return rest.Filter(must=cast(Any, list(must_conditions)))


def build_search_filter(
    *,
    query: SearchQuery,
    tenant_id: str,
    policy: VectorStorePolicy = DEFAULT_POLICY,
) -> rest.Filter:
    must_conditions: list[object] = [
        rest.FieldCondition(
            key=policy.tenant_id_field,
            match=rest.MatchValue(value=tenant_id),
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

    return rest.Filter(
        must=cast(Any, must_conditions),
        must_not=[
            rest.FieldCondition(
                key="_rag_kind", match=rest.MatchValue(value="head_v2")
            ),
            *(
                [
                    rest.FieldCondition(
                        key="_rag_revision",
                        match=rest.MatchAny(any=list(query.excluded_revisions)),
                    )
                ]
                if query.excluded_revisions
                else []
            ),
        ],
    )


def build_delete_filter(
    *,
    filter_values: DeleteFilter,
    tenant_id: str,
    policy: VectorStorePolicy = DEFAULT_POLICY,
) -> rest.Filter:
    return build_document_lookup_filter(
        tenant_id=tenant_id,
        collection=filter_values.collection,
        document_id=filter_values.document_id,
        document_key=None,
        policy=policy,
    )


def build_document_lookup_filter(
    *,
    tenant_id: str,
    collection: str | None,
    document_id: str | None,
    document_key: str | None,
    policy: VectorStorePolicy = DEFAULT_POLICY,
) -> rest.Filter:
    fields = (
        (policy.tenant_id_field, tenant_id),
        (policy.collection_field, collection),
        (policy.document_id_field, document_id),
        (policy.document_key_field, document_key),
    )
    return build_filter(
        [
            rest.FieldCondition(key=key, match=rest.MatchValue(value=value))
            for key, value in fields
            if value is not None
        ]
    )


def build_document_count_filter(
    *,
    tenant_id: str,
    collection: str,
    document_id: str,
    policy: VectorStorePolicy = DEFAULT_POLICY,
) -> rest.Filter:
    return build_document_lookup_filter(
        tenant_id=tenant_id,
        collection=collection,
        document_id=document_id,
        document_key=None,
        policy=policy,
    )
