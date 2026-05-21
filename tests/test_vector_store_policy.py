"""Coverage for ``VectorStorePolicy`` and its plumbing into filters and IDs.

The policy is the single source of truth for canonical payload-field names and
point-id format. These tests pin: (1) the default and custom point-id format,
and (2) that all four Qdrant filter builders honor the policy's renamed
fields.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable

import pytest
from qdrant_client import models as rest

from rag_core.search.indexer_points import make_point_id
from rag_core.search.policy import DEFAULT_POLICY, VectorStorePolicy
from rag_core.search.providers.qdrant_filters import (
    build_delete_filter,
    build_document_count_filter,
    build_document_lookup_filter,
    build_search_filter,
)
from rag_core.search.types import DeleteFilter, SearchQuery, SparseVector


def _expected_point_id(
    namespace: str,
    corpus_id: str,
    document_id: str,
    chunk_index: int,
) -> str:
    raw = f"{namespace.strip()}::{corpus_id.strip()}::{document_id}:chunk:{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


@pytest.mark.parametrize(
    "namespace,corpus_id,document_id,chunk_index",
    [
        ("team-space", "corpus-1", "doc-a", 0),
        (" team-space ", "corpus-1", "doc-a", 0),
        ("team-space", " corpus-1 ", "doc-a", 1),
        ("ns-xyz", "c-2", "doc-with-colons:and:more", 7),
    ],
)
def test_default_policy_make_point_id_matches_canonical_format(
    namespace: str, corpus_id: str, document_id: str, chunk_index: int
) -> None:
    expected = _expected_point_id(namespace, corpus_id, document_id, chunk_index)
    direct = make_point_id(
        namespace=namespace,
        corpus_id=corpus_id,
        document_id=document_id,
        chunk_index=chunk_index,
    )
    via_policy = DEFAULT_POLICY.make_point_id(
        namespace=namespace,
        corpus_id=corpus_id,
        document_id=document_id,
        chunk_index=chunk_index,
    )
    assert direct == expected
    assert via_policy == expected


def test_policy_make_point_id_honors_custom_format() -> None:
    custom = VectorStorePolicy(
        point_id_format=lambda ns, corpus, doc, idx: f"{ns}/{corpus}/{doc}/{idx}"
    )
    point_id = custom.make_point_id(
        namespace="ns",
        corpus_id="corpus",
        document_id="doc",
        chunk_index=3,
    )
    assert point_id == "ns/corpus/doc/3"


def _field_keys(qdrant_filter: rest.Filter) -> list[str]:
    return [
        condition.key
        for condition in (qdrant_filter.must or [])
        if isinstance(condition, rest.FieldCondition)
    ]


def _renamed_policy() -> VectorStorePolicy:
    return VectorStorePolicy(
        namespace_field="ns",
        corpus_id_field="cid",
        document_id_field="did",
        document_key_field="dkey",
        content_type_field="ct",
    )


def _search_filter_keys(policy: VectorStorePolicy) -> list[str]:
    query = SearchQuery(
        dense_vector=[0.1],
        sparse_vector=SparseVector(indices=[1], values=[1.0]),
        namespace="ns-1",
        corpus_ids=["c-1"],
        content_types=["document"],
        document_ids=["d-1"],
    )
    return _field_keys(build_search_filter(query=query, namespace="ns-1", policy=policy))


def _delete_filter_keys(policy: VectorStorePolicy) -> list[str]:
    return _field_keys(
        build_delete_filter(
            filter_values=DeleteFilter(
                namespace="ns-1", corpus_id="c-1", document_id="d-1"
            ),
            namespace="ns-1",
            policy=policy,
        )
    )


def _document_lookup_filter_keys(policy: VectorStorePolicy) -> list[str]:
    return _field_keys(
        build_document_lookup_filter(
            namespace="ns-1",
            corpus_id="c-1",
            document_id="d-1",
            document_key="/path/x",
            policy=policy,
        )
    )


def _document_count_filter_keys(policy: VectorStorePolicy) -> list[str]:
    return _field_keys(
        build_document_count_filter(
            namespace="ns-1",
            corpus_id="c-1",
            document_id="d-1",
            policy=policy,
        )
    )


@pytest.mark.parametrize(
    "builder,expected_keys",
    [
        (_search_filter_keys, ["ns", "cid", "ct", "did"]),
        (_delete_filter_keys, ["ns", "cid", "did"]),
        (_document_lookup_filter_keys, ["ns", "cid", "did", "dkey"]),
        (_document_count_filter_keys, ["ns", "cid", "did"]),
    ],
    ids=["search", "delete", "document_lookup", "document_count"],
)
def test_qdrant_filter_builders_use_policy_field_names(
    builder: Callable[[VectorStorePolicy], list[str]],
    expected_keys: list[str],
) -> None:
    assert builder(_renamed_policy()) == expected_keys
