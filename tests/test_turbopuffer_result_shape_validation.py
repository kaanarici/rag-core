from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from rag_core.search.providers.turbopuffer_store import TurboPufferVectorStore
from rag_core.search.request_models import SearchQuery
from rag_core.search.vector_models import (
    SparseVector,
    VectorPoint,
)

SECRET = "sk-test-secret"


class _SecretId:
    def __str__(self) -> str:
        return "point " + SECRET


class _MalformedNamespace:
    def __init__(
        self,
        *,
        rows: list[dict[str, object]] | None = None,
        aggregations: dict[str, object] | None = None,
        include_rows: bool = True,
    ) -> None:
        self.rows = rows
        self.aggregations = aggregations
        self.include_rows = include_rows
        self.write_calls: list[dict[str, object]] = []

    async def metadata(self) -> object:
        return SimpleNamespace()

    async def write(self, **kwargs: object) -> object:
        self.write_calls.append(kwargs)
        return SimpleNamespace(rows_remaining=False)

    async def query(self, **kwargs: object) -> object:
        response = SimpleNamespace()
        if self.include_rows:
            response.rows = list(self.rows or [])
        if self.aggregations is not None:
            response.aggregations = self.aggregations
        return response


def _store(namespace: _MalformedNamespace) -> TurboPufferVectorStore:
    return TurboPufferVectorStore(
        namespace="docs",
        dense_dimensions=3,
        namespace_client=namespace,
    )


def _query() -> SearchQuery:
    return SearchQuery(
        dense_vector=[1.0, 0.0, 0.0],
        sparse_vector=SparseVector(indices=[], values=[]),
        namespace="team-space",
        collections=["corpus-a"],
    )


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": "point-1",
        "$dist": 0.25,
        "namespace": "team-space",
        "collection": "corpus-a",
        "document_id": "doc-1",
        "document_key": "/doc.md",
        "content_sha256": "sha",
        "processing_version": "v1",
        "content_type": "document",
        "source_type": "file",
        "text": "alpha",
        "chunk_index": 0,
    }
    row.update(overrides)
    return row


def _point(point_id: str) -> VectorPoint:
    return VectorPoint(
        id=point_id,
        dense_vector=[1.0, 0.0, 0.0],
        sparse_vector=SparseVector(indices=[], values=[]),
        payload={
            "namespace": "team-space",
            "collection": "corpus-a",
            "document_id": "doc-1",
            "content_type": "document",
            "source_type": "file",
            "text": "alpha",
        },
    )


def test_turbopuffer_search_rejects_missing_response_rows() -> None:
    async def _run() -> None:
        store = _store(_MalformedNamespace(include_rows=False))

        with pytest.raises(ValueError) as exc_info:
            await store.search(_query())

        assert str(exc_info.value) == (
            "turbopuffer search response missing required rows"
        )

    asyncio.run(_run())


def test_turbopuffer_document_lookup_rejects_missing_response_rows() -> None:
    async def _run() -> None:
        store = _store(_MalformedNamespace(include_rows=False))

        with pytest.raises(ValueError) as exc_info:
            await store.get_document_record(
                namespace="team-space",
                collection="corpus-a",
                document_id="doc-1",
            )

        assert str(exc_info.value) == (
            "turbopuffer document lookup response missing required rows"
        )

    asyncio.run(_run())


def test_turbopuffer_search_rejects_missing_distance_without_leaking() -> None:
    async def _run() -> None:
        row = _row(text="private " + SECRET)
        del row["$dist"]
        store = _store(_MalformedNamespace(rows=[row]))

        with pytest.raises(ValueError) as exc_info:
            await store.search(_query())

        message = str(exc_info.value)
        assert message == "turbopuffer result row missing required field: $dist"
        assert SECRET not in message

    asyncio.run(_run())


@pytest.mark.parametrize(
    "distance",
    [
        pytest.param(True, id="bool"),
        pytest.param(-0.25, id="negative"),
        pytest.param("-0.25", id="negative-string"),
        pytest.param("nan", id="nan-string"),
        pytest.param(float("nan"), id="nan-float"),
        pytest.param(float("inf"), id="inf"),
        pytest.param(-float("inf"), id="negative-inf"),
        pytest.param(10**10000, id="huge-int"),
        pytest.param("not-a-number " + SECRET, id="secret-string"),
    ],
)
def test_turbopuffer_search_rejects_malformed_distances_without_leaking(
    distance: object,
) -> None:
    async def _run() -> None:
        store = _store(_MalformedNamespace(rows=[_row(**{"$dist": distance})]))

        with pytest.raises(ValueError) as exc_info:
            await store.search(_query())

        message = str(exc_info.value)
        assert message == "turbopuffer result row returned invalid field: $dist"
        assert SECRET not in message

    asyncio.run(_run())


def test_turbopuffer_search_reports_missing_row_id_safely() -> None:
    async def _run() -> None:
        row = _row(text="private " + SECRET)
        del row["id"]
        store = _store(_MalformedNamespace(rows=[row]))

        with pytest.raises(ValueError) as exc_info:
            await store.search(_query())

        message = str(exc_info.value)
        assert message == "turbopuffer result row missing required field: id"
        assert SECRET not in message

    asyncio.run(_run())


@pytest.mark.parametrize(
    "row_id",
    [
        pytest.param("", id="empty-string"),
        pytest.param("   ", id="blank-string"),
        pytest.param(True, id="bool"),
        pytest.param(123, id="int"),
        pytest.param(object(), id="object"),
        pytest.param(_SecretId(), id="secret-stringifier"),
    ],
)
def test_turbopuffer_search_reports_malformed_row_id_safely(row_id: object) -> None:
    async def _run() -> None:
        row = _row(id=row_id, text="private " + SECRET)
        store = _store(_MalformedNamespace(rows=[row]))

        with pytest.raises(ValueError) as exc_info:
            await store.search(_query())

        message = str(exc_info.value)
        assert message == "turbopuffer result row missing required field: id"
        assert SECRET not in message

    asyncio.run(_run())


def test_turbopuffer_search_preserves_valid_row_id() -> None:
    async def _run() -> None:
        store = _store(_MalformedNamespace(rows=[_row(id="point-valid")]))

        results = await store.search(_query())

        assert results[0].id == "point-valid"

    asyncio.run(_run())


@pytest.mark.parametrize("point_id", ["", "   "])
def test_turbopuffer_upsert_rejects_blank_point_ids(point_id: str) -> None:
    async def _run() -> None:
        namespace = _MalformedNamespace(rows=[])
        store = _store(namespace)

        with pytest.raises(ValueError) as exc_info:
            await store.upsert([_point(point_id)])

        assert str(exc_info.value) == "VectorPoint.id must be a non-empty string"
        assert namespace.write_calls == []

    asyncio.run(_run())


@pytest.mark.parametrize("point_id", ["x" * 65, "é" * 33])
def test_turbopuffer_upsert_rejects_point_ids_over_64_bytes(point_id: str) -> None:
    async def _run() -> None:
        namespace = _MalformedNamespace()
        store = _store(namespace)

        with pytest.raises(ValueError) as exc_info:
            await store.upsert([_point(point_id)])

        assert str(exc_info.value) == (
            "turbopuffer point id must be at most 64 UTF-8 bytes"
        )
        assert namespace.write_calls == []

    asyncio.run(_run())


def test_turbopuffer_upsert_rejects_unsupported_payload_object_without_leaking() -> None:
    async def _run() -> None:
        namespace = _MalformedNamespace()
        store = _store(namespace)
        point = VectorPoint(
            id="point-1",
            dense_vector=[1.0, 0.0, 0.0],
            sparse_vector=SparseVector(indices=[], values=[]),
            payload={"text": _SecretId()},
        )

        with pytest.raises(ValueError) as exc_info:
            await store.upsert([point])

        message = str(exc_info.value)
        assert message == (
            "vector payload contains unsupported value type: _SecretId"
        )
        assert SECRET not in message
        assert namespace.write_calls == []

    asyncio.run(_run())


@pytest.mark.parametrize("point_id", ["", "   "])
def test_turbopuffer_delete_point_ids_rejects_blank_ids(point_id: str) -> None:
    async def _run() -> None:
        namespace = _MalformedNamespace(rows=[])
        store = _store(namespace)

        with pytest.raises(ValueError) as exc_info:
            await store.delete_point_ids(["point-valid", point_id])

        assert str(exc_info.value) == "turbopuffer point id must be a non-empty string"
        assert namespace.write_calls == []

    asyncio.run(_run())


@pytest.mark.parametrize("point_id", ["x" * 65, "é" * 33])
def test_turbopuffer_delete_point_ids_rejects_ids_over_64_bytes(
    point_id: str,
) -> None:
    async def _run() -> None:
        namespace = _MalformedNamespace()
        store = _store(namespace)

        with pytest.raises(ValueError) as exc_info:
            await store.delete_point_ids(["point-valid", point_id])

        assert str(exc_info.value) == (
            "turbopuffer point id must be at most 64 UTF-8 bytes"
        )
        assert namespace.write_calls == []

    asyncio.run(_run())


@pytest.mark.parametrize(
    ("aggregations", "message"),
    [
        pytest.param(None, "missing required aggregation: chunk_count", id="missing"),
        pytest.param({}, "missing required aggregation: chunk_count", id="missing-key"),
        pytest.param(
            {"chunk_count": True},
            "returned invalid aggregation: chunk_count",
            id="bool",
        ),
        pytest.param(
            {"chunk_count": -1},
            "returned invalid aggregation: chunk_count",
            id="negative-int",
        ),
        pytest.param(
            {"chunk_count": "-1"},
            "returned invalid aggregation: chunk_count",
            id="negative-string",
        ),
        pytest.param(
            {"chunk_count": 1.5},
            "returned invalid aggregation: chunk_count",
            id="non-integer-float",
        ),
        pytest.param(
            {
                "chunk_count": "not-a-number\nTraceback (most recent call last): "
                + SECRET
            },
            "returned invalid aggregation: chunk_count",
            id="secret-string",
        ),
    ],
)
def test_turbopuffer_document_lookup_rejects_malformed_count_aggregation(
    aggregations: dict[str, object] | None,
    message: str,
) -> None:
    async def _run() -> None:
        namespace = _MalformedNamespace(
            rows=[_row(text="private " + SECRET)],
            aggregations=aggregations,
        )
        store = _store(namespace)

        with pytest.raises(ValueError) as exc_info:
            await store.get_document_record(
                namespace="team-space",
                collection="corpus-a",
                document_id="doc-1",
            )

        error = str(exc_info.value)
        assert message in error
        assert SECRET not in error
        assert "Traceback" not in error
        assert "private" not in error

    asyncio.run(_run())


@pytest.mark.parametrize(
    "field",
    ["document_id", "document_key", "content_sha256", "processing_version"],
)
def test_turbopuffer_document_lookup_rejects_non_string_record_fields(
    field: str,
) -> None:
    async def _run() -> None:
        namespace = _MalformedNamespace(
            rows=[_row(**{field: _SecretId(), "text": "private " + SECRET})],
            aggregations={"chunk_count": 1},
        )
        store = _store(namespace)

        with pytest.raises(ValueError) as exc_info:
            await store.get_document_record(
                namespace="team-space",
                collection="corpus-a",
                document_id="doc-1",
            )

        error = str(exc_info.value)
        assert error == (
            "turbopuffer document lookup returned invalid string field: " + field
        )
        assert SECRET not in error
        assert "private" not in error

    asyncio.run(_run())
