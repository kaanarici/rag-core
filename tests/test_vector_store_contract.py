"""Reusable behavior contract for first-party vector store adapters.

Each test runs against every first-party backend so the cross-backend behavior
contract stays single-sourced. Backend-specific edges live in their own files.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from dataclasses import dataclass

import pytest

from rag_core.search.providers.memory_store import InMemoryVectorStore
from rag_core.search.providers.qdrant_store import QdrantVectorStore
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

