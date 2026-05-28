"""Filter construction helpers for the Qdrant vector store.

Field names come from the ``VectorStorePolicy`` so adapters can override
payload conventions without rewriting filter logic.
"""

from __future__ import annotations

from typing import Any, Sequence, cast

from qdrant_client import models as rest

from rag_core.search.policy import DEFAULT_POLICY, VectorStorePolicy
from rag_core.search.request_models import (
    DeleteFilter,
    SearchQuery,
)

from .qdrant_metadata_filters import metadata_filter_to_qdrant


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

    if query.corpus_ids:
        must_conditions.append(
            rest.FieldCondition(
                key=policy.corpus_id_field,
                match=rest.MatchAny(any=query.corpus_ids),
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

    if filter_values.corpus_id:
        must_conditions.append(
            rest.FieldCondition(
                key=policy.corpus_id_field,
                match=rest.MatchValue(value=filter_values.corpus_id),
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
    corpus_id: str,
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
            key=policy.corpus_id_field,
            match=rest.MatchValue(value=corpus_id),
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
    corpus_id: str,
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
                key=policy.corpus_id_field,
                match=rest.MatchValue(value=corpus_id),
            ),
            rest.FieldCondition(
                key=policy.document_id_field,
                match=rest.MatchValue(value=document_id),
            ),
        ]
    )
