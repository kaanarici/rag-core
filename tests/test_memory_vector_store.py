"""Behavior that the in-memory store implements on its own.

The cross-vector-store contract lives in ``test_vector_store_contract.py``; these
tests cover memory-store specifics: payload-driven filtering (corpus,
document, content type), sparse-only search ranking, namespace input
validation, missing-record fallthrough, and that ``close`` resets state.
"""

from __future__ import annotations

import asyncio

import pytest

from rag_core.search.planning import query_plan_preset
from rag_core.search.providers.memory_store import InMemoryVectorStore
from rag_core.search.request_models import SearchQuery
from rag_core.search.vector_models import (
    ContentType,
    SparseVector,
    VectorPoint,
)


def _make_point(
    *,
    point_id: str,
    namespace: str = "team-space",
    collection: str = "corpus-a",
    document_id: str = "doc-1",
    content_type: object = "document",
    dense: list[float] | None = None,
    sparse_indices: list[int] | None = None,
    sparse_values: list[float] | None = None,
    text: str = "hello world",
    chunk_index: int = 0,
) -> VectorPoint:
    return VectorPoint(
        id=point_id,
        dense_vector=dense or [1.0, 0.0, 0.0],
        sparse_vector=SparseVector(
            indices=sparse_indices or [1, 2],
            values=sparse_values or [1.0, 1.0],
        ),
        payload={
            "namespace": namespace,
            "collection": collection,
            "document_id": document_id,
            "content_type": content_type,
            "source_type": "file",
            "text": text,
            "chunk_index": chunk_index,
        },
    )


def _empty_sparse() -> SparseVector:
    return SparseVector(indices=[], values=[])


def test_memory_store_sparse_only_search_finds_matching_points() -> None:
    async def _run() -> None:
        store = InMemoryVectorStore()
        await store.upsert(
            [
                _make_point(
                    point_id="p1",
                    sparse_indices=[10, 20],
                    sparse_values=[1.0, 1.0],
                    text="a",
                ),
                _make_point(
                    point_id="p2",
                    sparse_indices=[30, 40],
                    sparse_values=[1.0, 1.0],
                    text="b",
                ),
            ]
        )

        results = await store.search(
            SearchQuery(
                dense_vector=[],
                sparse_vector=SparseVector(indices=[10], values=[1.0]),
                namespace="team-space",
                collections=["corpus-a"],
                limit=2,
            )
        )
        assert [hit.id for hit in results] == ["p1"]

    asyncio.run(_run())


@pytest.mark.parametrize(
    "label,points,query_kwargs,expected_ids",
    [
        (
            "namespace_and_corpus",
            [
                {"point_id": "p1", "namespace": "team-a", "collection": "corpus-1"},
                {"point_id": "p2", "namespace": "team-b", "collection": "corpus-1"},
                {"point_id": "p3", "namespace": "team-a", "collection": "corpus-2"},
            ],
            {"namespace": "team-a", "collections": ["corpus-1"]},
            ["p1"],
        ),
        (
            "document_ids",
            [
                {"point_id": "p1", "document_id": "doc-x"},
                {"point_id": "p2", "document_id": "doc-y"},
            ],
            {
                "namespace": "team-space",
                "collections": ["corpus-a"],
                "document_ids": ["doc-y"],
            },
            ["p2"],
        ),
        (
            "content_types",
            [
                {"point_id": "p1", "content_type": "document"},
                {"point_id": "p2", "content_type": "code"},
            ],
            {
                "namespace": "team-space",
                "collections": ["corpus-a"],
                "content_types": ["code"],
            },
            ["p2"],
        ),
        (
            "content_type_enum_payload",
            [
                {"point_id": "p1", "content_type": ContentType.DOCUMENT},
                {"point_id": "p2", "content_type": ContentType.CODE},
            ],
            {
                "namespace": "team-space",
                "collections": ["corpus-a"],
                "content_types": ["document"],
            },
            ["p1"],
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_memory_store_search_applies_payload_filters(
    label: str,
    points: list[dict[str, object]],
    query_kwargs: dict[str, object],
    expected_ids: list[str],
) -> None:
    del label  # only used as the test id

    async def _run() -> None:
        store = InMemoryVectorStore()
        await store.upsert([_make_point(**point) for point in points])  # type: ignore[arg-type]

        results = await store.search(
            SearchQuery(
                dense_vector=[1.0, 0.0, 0.0],
                sparse_vector=_empty_sparse(),
                limit=10,
                **query_kwargs,  # type: ignore[arg-type]
            )
        )
        assert [hit.id for hit in results] == expected_ids

    asyncio.run(_run())


def test_memory_store_search_requires_namespace() -> None:
    async def _run() -> None:
        with pytest.raises(
            ValueError,
            match="SearchQuery.namespace must be a non-empty string",
        ):
            SearchQuery(
                dense_vector=[1.0],
                sparse_vector=_empty_sparse(),
                namespace=" ",
                collections=["corpus-a"],
            )

    asyncio.run(_run())


def test_memory_store_get_document_record_returns_none_when_missing() -> None:
    async def _run() -> None:
        store = InMemoryVectorStore()
        record = await store.get_document_record(
            namespace="team-space",
            collection="corpus-a",
            document_id="doc-missing",
        )
        assert record is None

    asyncio.run(_run())


def test_memory_store_check_health_reports_adapter_and_point_count() -> None:
    async def _run() -> None:
        store = InMemoryVectorStore()
        await store.upsert([_make_point(point_id="p1")])
        health = await store.check_health()
        assert health["healthy"] is True
        assert health["adapter"] == "memory"
        assert "backend" not in health
        assert health["points_count"] == 1

    asyncio.run(_run())


def test_memory_store_rejects_mismatched_dense_dimensions() -> None:
    async def _run() -> None:
        store = InMemoryVectorStore()
        await store.upsert([_make_point(point_id="p1", dense=[1.0, 0.0, 0.0])])
        assert store.capabilities.dense_vector_dimensions == 3

        with pytest.raises(ValueError, match="expected 3 dimensions, got 2"):
            await store.upsert([_make_point(point_id="p2", dense=[1.0, 0.0])])

        with pytest.raises(ValueError, match="expected 3 dimensions, got 2"):
            await store.search(
                SearchQuery(
                    dense_vector=[1.0, 0.0],
                    sparse_vector=_empty_sparse(),
                    namespace="team-space",
                    collections=["corpus-a"],
                    limit=1,
                )
            )

    asyncio.run(_run())


def test_memory_store_sparse_only_plan_ignores_unused_dense_query_dimensions() -> None:
    async def _run() -> None:
        store = InMemoryVectorStore()
        await store.upsert([_make_point(point_id="p1", dense=[1.0, 0.0, 0.0])])

        results = await store.search(
            SearchQuery(
                dense_vector=[1.0, 0.0],
                sparse_vector=SparseVector(indices=[1], values=[1.0]),
                namespace="team-space",
                collections=["corpus-a"],
                limit=1,
                query_plan=query_plan_preset("sparse_only", limit=1),
            )
        )

        assert [hit.id for hit in results] == ["p1"]

    asyncio.run(_run())


def test_memory_store_default_search_does_not_reuse_bm25_for_missing_sparse_channels() -> None:
    async def _run() -> None:
        store = InMemoryVectorStore()
        await store.upsert(
            [
                VectorPoint(
                    id="bm25-only",
                    dense_vector=[0.0, 0.0, 0.0],
                    sparse_vector=SparseVector(indices=[1], values=[1.0]),
                    sparse_vectors={},
                    payload={
                        "namespace": "team-space",
                        "collection": "corpus-a",
                        "document_id": "doc-1",
                        "content_type": "document",
                        "source_type": "file",
                        "text": "bm25 only",
                        "chunk_index": 0,
                    },
                )
            ]
        )

        results = await store.search(
            SearchQuery(
                dense_vector=[],
                sparse_vector=SparseVector(indices=[9], values=[1.0]),
                sparse_vectors={"splade": SparseVector(indices=[1], values=[1.0])},
                namespace="team-space",
                collections=["corpus-a"],
                limit=1,
            )
        )

        assert results == []

    asyncio.run(_run())


def test_memory_store_dense_search_skips_points_without_dense_vectors() -> None:
    async def _run() -> None:
        store = InMemoryVectorStore()
        await store.upsert(
            [
                _make_point(point_id="dense", dense=[1.0, 0.0, 0.0], text="dense"),
                VectorPoint(
                    id="sparse-only",
                    dense_vector=[],
                    sparse_vector=SparseVector(indices=[1], values=[1.0]),
                    payload={
                        "namespace": "team-space",
                        "collection": "corpus-a",
                        "document_id": "doc-sparse",
                        "content_type": "document",
                        "source_type": "file",
                        "text": "sparse only",
                        "chunk_index": 0,
                    },
                ),
            ]
        )

        results = await store.search(
            SearchQuery(
                dense_vector=[1.0, 0.0, 0.0],
                sparse_vector=_empty_sparse(),
                namespace="team-space",
                collections=["corpus-a"],
                limit=5,
                query_plan=query_plan_preset("dense_only", limit=5),
            )
        )

        assert [hit.id for hit in results] == ["dense"]

    asyncio.run(_run())


def test_memory_store_document_record_rejects_malformed_identifier_payload() -> None:
    async def _run() -> None:
        store = InMemoryVectorStore()
        await store.upsert(
            [
                VectorPoint(
                    id="p1",
                    dense_vector=[1.0, 0.0, 0.0],
                    sparse_vector=SparseVector(indices=[1], values=[1.0]),
                    payload={
                        "namespace": "team-space",
                        "collection": "corpus-a",
                        "document_id": 123,
                        "document_key": "/docs/bad.md",
                        "content_type": "document",
                        "source_type": "file",
                        "text": "bad id payload",
                        "chunk_index": 0,
                    },
                )
            ]
        )

        with pytest.raises(
            ValueError,
            match="payload field 'document_id' must be a string",
        ):
            await store.get_document_record(
                namespace="team-space",
                collection="corpus-a",
                document_key="/docs/bad.md",
            )

    asyncio.run(_run())


def test_memory_store_close_drops_indexed_points() -> None:
    async def _run() -> None:
        store = InMemoryVectorStore()
        await store.upsert([_make_point(point_id="p1")])
        await store.close()
        health = await store.check_health()
        assert health["points_count"] == 0

    asyncio.run(_run())
