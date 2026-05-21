"""Reusable behavior contract for first-party vector store adapters.

Each test runs against every first-party backend so the cross-backend behavior
contract stays single-sourced. Backend-specific edges live in their own files.
"""

from __future__ import annotations

import asyncio
import math
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import SupportsFloat, SupportsIndex

import pytest

from rag_core.search.providers.memory_store import InMemoryVectorStore
from rag_core.search.providers.qdrant_store import QdrantVectorStore
from rag_core.search.providers.turbopuffer_store import TurboPufferVectorStore
from rag_core.search.types import (
    DeleteFilter,
    SearchQuery,
    SparseVector,
    Term,
    VectorPoint,
    VectorStore,
)


@dataclass(frozen=True)
class _VectorStoreCase:
    name: str
    create: Callable[[], VectorStore]
    dense_vector_dimensions: int | None


def _store_cases() -> list[_VectorStoreCase]:
    return [
        _VectorStoreCase(
            name="memory",
            create=InMemoryVectorStore,
            dense_vector_dimensions=None,
        ),
        _VectorStoreCase(
            name="qdrant-local",
            create=lambda: QdrantVectorStore(
                url=None,
                api_key=None,
                location=":memory:",
                collection_name=f"contract_{uuid.uuid4().hex}",
                dense_dimensions=3,
                quantization_enabled=False,
            ),
            dense_vector_dimensions=3,
        ),
        _VectorStoreCase(
            name="turbopuffer-fake",
            create=lambda: TurboPufferVectorStore(
                namespace=f"contract-{uuid.uuid4().hex}",
                dense_dimensions=3,
                namespace_client=_FakeTurboPufferNamespace(),
            ),
            dense_vector_dimensions=3,
        ),
    ]


@pytest.fixture(params=_store_cases(), ids=lambda case: case.name)
def case(request: pytest.FixtureRequest) -> _VectorStoreCase:
    return request.param  # type: ignore[no-any-return]


def test_vector_store_contract_dense_search_and_scoping(case: _VectorStoreCase) -> None:
    async def _run_test() -> None:
        store = case.create()
        try:
            # ensure_collection is idempotent.
            await store.ensure_collection()
            await store.ensure_collection()
            points = _contract_points()
            await store.upsert(points)

            results = await store.search(
                SearchQuery(
                    dense_vector=[1.0, 0.0, 0.0],
                    sparse_vector=_empty_sparse(),
                    namespace="contract-namespace",
                    corpus_ids=["corpus-alpha"],
                    limit=10,
                )
            )

            assert {hit.document_id for hit in results} == {"doc-alpha"}
            assert results[0].id == points[0].id
            assert results[0].text == "alpha opening"
            assert results[0].corpus_id == "corpus-alpha"
            assert results[0].content_type == "document"
        finally:
            await store.close()

    asyncio.run(_run_test())


def test_vector_store_contract_metadata_filters(case: _VectorStoreCase) -> None:
    async def _run_test() -> None:
        store = case.create()
        try:
            await store.upsert(_contract_points())

            results = await store.search(
                SearchQuery(
                    dense_vector=[1.0, 0.0, 0.0],
                    sparse_vector=_empty_sparse(),
                    namespace="contract-namespace",
                    corpus_ids=["corpus-alpha"],
                    limit=10,
                    metadata_filter=Term(field="section", value="details"),
                )
            )

            assert [hit.metadata["section"] for hit in results] == ["details"]
            assert results[0].text == "alpha details"
        finally:
            await store.close()

    asyncio.run(_run_test())


def test_vector_store_contract_document_record_lookup(case: _VectorStoreCase) -> None:
    async def _run_test() -> None:
        store = case.create()
        try:
            assert store.capabilities.per_point_delete is True
            assert store.capabilities.document_record_lookup is True
            assert (
                store.capabilities.dense_vector_dimensions
                == case.dense_vector_dimensions
            )
            await store.upsert(_contract_points())

            by_id = await store.get_document_record(
                namespace="contract-namespace",
                corpus_id="corpus-alpha",
                document_id="doc-alpha",
            )
            by_key = await store.get_document_record(
                namespace="contract-namespace",
                corpus_id="corpus-alpha",
                document_key="/docs/alpha.md",
            )

            assert by_id is not None
            assert by_key == by_id
            assert by_id.document_id == "doc-alpha"
            assert by_id.document_key == "/docs/alpha.md"
            assert by_id.content_sha256 == "sha-alpha"
            assert by_id.processing_version == "v1"
            assert by_id.chunk_count == 2
        finally:
            await store.close()

    asyncio.run(_run_test())


def test_vector_store_contract_key_lookup_rejects_missing_document_id(
    case: _VectorStoreCase,
) -> None:
    async def _run_test() -> None:
        store = case.create()
        try:
            points = _contract_points()
            await store.upsert(
                [
                    _point_without_document_id(
                        points[0],
                        point_id=str(uuid.uuid4()),
                        document_key="/docs/missing-a.md",
                    ),
                    _point_without_document_id(
                        points[1],
                        point_id=str(uuid.uuid4()),
                        document_key="/docs/missing-b.md",
                    ),
                ]
            )

            record = await store.get_document_record(
                namespace="contract-namespace",
                corpus_id="corpus-alpha",
                document_key="/docs/missing-a.md",
            )

            assert record is None
        finally:
            await store.close()

    asyncio.run(_run_test())


def test_vector_store_contract_delete_by_document_scope(case: _VectorStoreCase) -> None:
    async def _run_test() -> None:
        store = case.create()
        try:
            await store.upsert(_contract_points())

            await store.delete(
                DeleteFilter(
                    namespace="contract-namespace",
                    corpus_id="corpus-alpha",
                    document_id="doc-alpha",
                )
            )

            deleted = await store.search(
                SearchQuery(
                    dense_vector=[1.0, 0.0, 0.0],
                    sparse_vector=_empty_sparse(),
                    namespace="contract-namespace",
                    corpus_ids=["corpus-alpha"],
                    limit=10,
                )
            )
            preserved = await store.search(
                SearchQuery(
                    dense_vector=[0.0, 1.0, 0.0],
                    sparse_vector=_empty_sparse(),
                    namespace="contract-namespace",
                    corpus_ids=["corpus-beta"],
                    limit=10,
                )
            )

            assert deleted == []
            assert [hit.document_id for hit in preserved] == ["doc-beta"]
        finally:
            await store.close()

    asyncio.run(_run_test())


def test_vector_store_contract_delete_point_ids(case: _VectorStoreCase) -> None:
    async def _run_test() -> None:
        store = case.create()
        try:
            points = _contract_points()
            await store.upsert(points)

            await store.delete_point_ids([points[0].id])

            results = await store.search(
                SearchQuery(
                    dense_vector=[1.0, 0.0, 0.0],
                    sparse_vector=_empty_sparse(),
                    namespace="contract-namespace",
                    corpus_ids=["corpus-alpha"],
                    limit=10,
                )
            )

            assert [hit.id for hit in results] == [points[1].id]
            assert results[0].text == "alpha details"
        finally:
            await store.close()

    asyncio.run(_run_test())


def test_vector_store_contract_health(case: _VectorStoreCase) -> None:
    async def _run_test() -> None:
        store = case.create()
        try:
            await store.ensure_collection()
            health = await store.check_health()

            assert health["healthy"] is True
        finally:
            await store.close()

    asyncio.run(_run_test())


def _contract_points() -> list[VectorPoint]:
    point_ids = [str(uuid.uuid4()) for _ in range(3)]
    return [
        _point(
            point_id=point_ids[0],
            dense=[1.0, 0.0, 0.0],
            corpus_id="corpus-alpha",
            document_id="doc-alpha",
            document_key="/docs/alpha.md",
            text="alpha opening",
            chunk_index=0,
            section="opening",
        ),
        _point(
            point_id=point_ids[1],
            dense=[0.95, 0.05, 0.0],
            corpus_id="corpus-alpha",
            document_id="doc-alpha",
            document_key="/docs/alpha.md",
            text="alpha details",
            chunk_index=1,
            section="details",
        ),
        _point(
            point_id=point_ids[2],
            dense=[0.0, 1.0, 0.0],
            corpus_id="corpus-beta",
            document_id="doc-beta",
            document_key="/docs/beta.md",
            text="beta appendix",
            chunk_index=0,
            section="appendix",
            content_sha256="sha-beta",
        ),
    ]


def _point(
    *,
    point_id: str,
    dense: list[float],
    corpus_id: str,
    document_id: str,
    document_key: str,
    text: str,
    chunk_index: int,
    section: str,
    content_sha256: str = "sha-alpha",
) -> VectorPoint:
    return VectorPoint(
        id=point_id,
        dense_vector=dense,
        sparse_vector=_empty_sparse(),
        payload={
            "namespace": "contract-namespace",
            "corpus_id": corpus_id,
            "document_id": document_id,
            "document_key": document_key,
            "content_sha256": content_sha256,
            "processing_version": "v1",
            "content_type": "document",
            "source_type": "file",
            "text": text,
            "chunk_index": chunk_index,
            "section": section,
        },
    )


def _point_without_document_id(
    point: VectorPoint,
    *,
    point_id: str,
    document_key: str,
) -> VectorPoint:
    payload = dict(point.payload)
    payload.pop("document_id", None)
    payload["document_key"] = document_key
    return VectorPoint(
        id=point_id,
        dense_vector=point.dense_vector,
        sparse_vector=point.sparse_vector,
        payload=payload,
    )


def _empty_sparse() -> SparseVector:
    return SparseVector(indices=[], values=[])


class _FakeTurboPufferNamespace:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, object]] = {}

    async def metadata(self) -> object:
        return SimpleNamespace(
            approx_row_count=len(self.rows),
            approx_logical_bytes=0,
            index=SimpleNamespace(status="up-to-date"),
        )

    async def write(self, **kwargs: object) -> object:
        for row in _object_list(kwargs.get("upsert_rows")):
            assert isinstance(row, dict)
            self.rows[str(row["id"])] = dict(row)
        for point_id in _object_list(kwargs.get("deletes")):
            self.rows.pop(str(point_id), None)
        delete_filter = kwargs.get("delete_by_filter")
        if delete_filter is not None:
            self.rows = {
                point_id: row
                for point_id, row in self.rows.items()
                if not _matches_turbopuffer_filter(row, delete_filter)
            }
        return SimpleNamespace(rows_remaining=False)

    async def query(self, **kwargs: object) -> object:
        filters = kwargs.get("filters")
        rows = [
            row
            for row in self.rows.values()
            if filters is None or _matches_turbopuffer_filter(row, filters)
        ]
        rank_by = kwargs.get("rank_by")
        if isinstance(rank_by, (list, tuple)) and tuple(rank_by[:2]) == (
            "vector",
            "ANN",
        ):
            query_vector = rank_by[2]
            assert isinstance(query_vector, list)
            rows = sorted(
                rows,
                key=lambda row: _cosine_distance(
                    query_vector,
                    row.get("vector") if isinstance(row.get("vector"), list) else [],
                ),
            )
            rows = [
                {
                    **row,
                    "$dist": _cosine_distance(
                        query_vector,
                        row.get("vector")
                        if isinstance(row.get("vector"), list)
                        else [],
                    ),
                }
                for row in rows
            ]
        elif isinstance(rank_by, (list, tuple)) and tuple(rank_by) == (
            "chunk_index",
            "asc",
        ):
            rows = sorted(rows, key=lambda row: _int_value(row.get("chunk_index"), 0))
        limit = _int_value(kwargs.get("top_k") or kwargs.get("limit"), len(rows))
        aggregations = None
        if kwargs.get("aggregate_by") is not None:
            aggregations = {"chunk_count": len(rows)}
        return SimpleNamespace(rows=rows[:limit], aggregations=aggregations)


def _matches_turbopuffer_filter(row: dict[str, object], filter_value: object) -> bool:
    assert isinstance(filter_value, (list, tuple))
    op = filter_value[0]
    if op == "And":
        children = filter_value[1]
        assert isinstance(children, (list, tuple))
        return all(_matches_turbopuffer_filter(row, child) for child in children)
    if op == "Or":
        children = filter_value[1]
        assert isinstance(children, (list, tuple))
        return any(_matches_turbopuffer_filter(row, child) for child in children)
    if op == "Not":
        return not _matches_turbopuffer_filter(row, filter_value[1])

    assert len(filter_value) == 3
    field, operator, expected = filter_value
    actual = row.get(str(field))
    if operator == "Eq":
        return bool(actual == expected)
    if operator == "In":
        assert isinstance(expected, (list, tuple, set))
        return actual in expected
    if operator == "Gte":
        return _float_value(actual) >= _float_value(expected)
    if operator == "Gt":
        return _float_value(actual) > _float_value(expected)
    if operator == "Lte":
        return _float_value(actual) <= _float_value(expected)
    if operator == "Lt":
        return _float_value(actual) < _float_value(expected)
    raise AssertionError(f"unsupported fake filter operator: {operator}")


def _cosine_distance(query: list[float], vector: object) -> float:
    if not isinstance(vector, list):
        return 1.0
    numeric_vector = [float(value) for value in vector]
    dot = sum(a * b for a, b in zip(query, numeric_vector, strict=False))
    query_norm = math.sqrt(sum(value * value for value in query))
    vector_norm = math.sqrt(sum(value * value for value in numeric_vector))
    if query_norm == 0.0 or vector_norm == 0.0:
        return 1.0
    return 1.0 - (dot / (query_norm * vector_norm))


def _object_list(value: object) -> list[object]:
    if value is None:
        return []
    assert isinstance(value, list)
    return value


def _int_value(value: object, fallback: int) -> int:
    if value is None:
        return fallback
    if isinstance(value, (str, bytes, bytearray, SupportsIndex)):
        return int(value)
    return fallback


def _float_value(value: object) -> float:
    assert isinstance(value, (str, bytes, bytearray, SupportsFloat, SupportsIndex))
    return float(value)
