"""TurboPuffer-specific adapter behavior.

The shared retrieve/delete/upsert contract is covered by
``test_vector_store_contract.py`` against the same fake namespace. These tests
pin the TurboPuffer-only wire shape: schema, distance metric, filter tuple
translation, count-aggregation document lookup, dense-dimension preflight,
and the metadata-derived health payload.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from types import SimpleNamespace
from typing import cast

import pytest

from rag_core.search.policy import DEFAULT_POLICY
from rag_core.search.pipeline import HybridRetrieve, PipelineContext, PipelineQuery
from rag_core.search.providers.turbopuffer_store import (
    TurboPufferDeleteByFilterExhausted,
    TurboPufferVectorStore,
    delete_turbopuffer_filter,
)
from rag_core.search.providers.turbopuffer_payloads import TURBOPUFFER_BM25_TEXT_FIELD
from rag_core.search.filters import Range
from rag_core.search.provider_protocols import MetadataFilterCapabilities
from rag_core.search.request_models import (
    DeleteFilter,
    SearchQuery,
)
from rag_core.search.vector_models import (
    SparseVector,
    VectorPoint,
)
from rag_core.search.planning import query_plan_preset
from tests.support import FakeEmbeddingProvider, FakeSparseEmbedder
from tests.support.turbopuffer_fake import TurboPufferFakeNamespace


class _RecordingNamespace:
    def __init__(self, *, rows_remaining: list[object] | None = None) -> None:
        self.write_calls: list[dict[str, object]] = []
        self.query_calls: list[dict[str, object]] = []
        self._rows_remaining = list(rows_remaining or [])

    async def metadata(self) -> object:
        return SimpleNamespace(
            approx_row_count=3,
            approx_logical_bytes=99,
            index=SimpleNamespace(status="up-to-date"),
        )

    async def write(self, **kwargs: object) -> object:
        self.write_calls.append(kwargs)
        rows_remaining = self._rows_remaining.pop(0) if self._rows_remaining else False
        return SimpleNamespace(rows_remaining=rows_remaining)

    async def query(self, **kwargs: object) -> object:
        self.query_calls.append(kwargs)
        response = SimpleNamespace(
            rows=[
                {
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
            ]
        )
        if "aggregate_by" in kwargs:
            response.aggregations = {"chunk_count": 7}
        return response


class _FailingSparseEmbedder(FakeSparseEmbedder):
    def embed_query_multi(self, query: str) -> dict[str, SparseVector]:
        self.embed_query_multi_calls.append(query)
        raise RuntimeError("sparse query vectors should not be required")


class _DuplicateDenseTopHitNamespace(TurboPufferFakeNamespace):
    async def query(self, **kwargs: object) -> object:
        response = await super().query(**kwargs)
        rank_by = kwargs.get("rank_by")
        if (
            isinstance(rank_by, tuple)
            and len(rank_by) == 3
            and rank_by[0] == "vector"
            and rank_by[1] == "ANN"
        ):
            rows = getattr(response, "rows", None)
            if isinstance(rows, list) and rows and isinstance(rows[0], dict):
                setattr(response, "rows", [dict(rows[0]), dict(rows[0]), *rows[1:]])
        return response


def _store(namespace: object, *, dense_dimensions: int = 3) -> TurboPufferVectorStore:
    return TurboPufferVectorStore(
        namespace="docs",
        dense_dimensions=dense_dimensions,
        namespace_client=namespace,
    )


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
            "text": point_id,
        },
    )


def _store_with_metric(
    namespace: _RecordingNamespace,
    *,
    distance_metric: str,
) -> TurboPufferVectorStore:
    return TurboPufferVectorStore(
        namespace="docs",
        dense_dimensions=3,
        distance_metric=distance_metric,
        namespace_client=namespace,
    )


def test_turbopuffer_declares_metadata_filter_capabilities() -> None:
    store = _store(_RecordingNamespace())
    assert store.capabilities.metadata_filter == MetadataFilterCapabilities(
        term=True,
        in_=True,
        numeric_range=True,
        string_range=True,
        geo=False,
        boolean=True,
    )


@pytest.mark.parametrize("namespace", ["has/slash", "has space", "x" * 129])
def test_turbopuffer_store_rejects_invalid_physical_namespace(namespace: str) -> None:
    with pytest.raises(
        ValueError,
        match=r"TurboPufferVectorStore namespace must match "
        r"\[A-Za-z0-9-_\.\]\{1,128\}",
    ):
        TurboPufferVectorStore(
            namespace=namespace,
            dense_dimensions=3,
            namespace_client=_RecordingNamespace(),
        )


def test_turbopuffer_store_writes_rows_schema_and_distance_metric() -> None:
    async def _run() -> None:
        namespace = _RecordingNamespace()
        store = _store(namespace)
        await store.upsert([_point("point-1")])

        call = namespace.write_calls[0]
        rows = call["upsert_rows"]
        assert isinstance(rows, list)
        assert rows[0]["id"] == "point-1"
        assert rows[0]["vector"] == [1.0, 0.0, 0.0]
        assert "sparse_vector" not in rows[0]
        assert call["distance_metric"] == "cosine_distance"
        schema = call["schema"]
        assert isinstance(schema, dict)
        assert schema["vector"] == {"type": "[3]f32", "ann": True}
        assert "sparse_vector" not in schema
        assert schema["text"]["filterable"] is False
        assert schema["text"]["full_text_search"] is False
        assert schema[TURBOPUFFER_BM25_TEXT_FIELD] == {
            "type": "string",
            "filterable": False,
            "full_text_search": True,
        }
        assert rows[0][TURBOPUFFER_BM25_TEXT_FIELD] == "point-1"

    asyncio.run(_run())


def test_turbopuffer_store_does_not_serialize_unqueried_sparse_vectors() -> None:
    async def _run() -> None:
        namespace = _RecordingNamespace()
        store = _store(namespace)
        await store.upsert(
            [
                VectorPoint(
                    id="sparse-point",
                    dense_vector=[1.0, 0.0, 0.0],
                    sparse_vector=SparseVector(indices=[0, 2], values=[0.9, 0.1]),
                    payload={
                        "namespace": "team-space",
                        "collection": "corpus-a",
                        "document_id": "doc-1",
                        "content_type": "document",
                        "source_type": "file",
                        "text": "sparse payload",
                    },
                )
            ]
        )

        rows = cast(list[dict[str, object]], namespace.write_calls[0]["upsert_rows"])
        assert "sparse_vector" not in rows[0]

    asyncio.run(_run())


def test_turbopuffer_store_hybrid_rrf_queries_dense_and_bm25_then_fuses() -> None:
    async def _run() -> None:
        namespace = TurboPufferFakeNamespace()
        store = _store(namespace)
        await store.upsert(
            [
                VectorPoint(
                    id="alpha",
                    dense_vector=[1.0, 0.0, 0.0],
                    sparse_vector=SparseVector(indices=[], values=[]),
                    payload={
                        "namespace": "team-space",
                        "collection": "corpus-a",
                        "document_id": "doc-alpha",
                        "content_type": "document",
                        "source_type": "file",
                        "text": "semantic only",
                    },
                ),
                VectorPoint(
                    id="beta",
                    dense_vector=[0.0, 1.0, 0.0],
                    sparse_vector=SparseVector(indices=[], values=[]),
                    payload={
                        "namespace": "team-space",
                        "collection": "corpus-a",
                        "document_id": "doc-beta",
                        "content_type": "document",
                        "source_type": "file",
                        "text": "fox jumping fox",
                    },
                ),
                VectorPoint(
                    id="gamma",
                    dense_vector=[0.9, 0.0, 0.1],
                    sparse_vector=SparseVector(indices=[], values=[]),
                    payload={
                        "namespace": "team-space",
                        "collection": "corpus-a",
                        "document_id": "doc-gamma",
                        "content_type": "document",
                        "source_type": "file",
                        "text": "fox dense",
                    },
                ),
            ]
        )

        results = await store.search(
            SearchQuery(
                dense_vector=[1.0, 0.0, 0.0],
                sparse_vector=SparseVector(indices=[1], values=[1.0]),
                namespace="team-space",
                collections=["corpus-a"],
                limit=2,
                lexical_query="fox",
                query_plan=query_plan_preset("hybrid_rrf", limit=2),
            )
        )

        rank_by_values = [call["rank_by"] for call in namespace.query_calls]
        assert ("vector", "ANN", [1.0, 0.0, 0.0]) in rank_by_values
        assert (TURBOPUFFER_BM25_TEXT_FIELD, "BM25", "fox") in rank_by_values
        assert all(call["top_k"] == 20 for call in namespace.query_calls)
        assert [result.document_id for result in results] == ["doc-beta", "doc-gamma"]
        assert results[0].score > results[1].score

    asyncio.run(_run())


def test_turbopuffer_hybrid_rrf_dedupes_duplicate_ids_within_dense_list() -> None:
    async def _run() -> None:
        namespace = _DuplicateDenseTopHitNamespace()
        store = _store(namespace)
        await store.upsert(
            [
                VectorPoint(
                    id="alpha",
                    dense_vector=[1.0, 0.0, 0.0],
                    sparse_vector=SparseVector(indices=[], values=[]),
                    payload={
                        "namespace": "team-space",
                        "collection": "corpus-a",
                        "document_id": "doc-alpha",
                        "content_type": "document",
                        "source_type": "file",
                        "text": "semantic only",
                    },
                ),
                VectorPoint(
                    id="beta",
                    dense_vector=[0.8, 0.6, 0.0],
                    sparse_vector=SparseVector(indices=[], values=[]),
                    payload={
                        "namespace": "team-space",
                        "collection": "corpus-a",
                        "document_id": "doc-beta",
                        "content_type": "document",
                        "source_type": "file",
                        "text": "fox lexical",
                    },
                ),
            ]
        )

        results = await store.search(
            SearchQuery(
                dense_vector=[1.0, 0.0, 0.0],
                sparse_vector=SparseVector(indices=[1], values=[1.0]),
                namespace="team-space",
                collections=["corpus-a"],
                limit=2,
                lexical_query="fox",
                query_plan=query_plan_preset("hybrid_rrf", limit=2),
            )
        )

        assert [result.document_id for result in results] == ["doc-beta", "doc-alpha"]
        assert results[0].score > results[1].score

    asyncio.run(_run())


def test_turbopuffer_hybrid_pipeline_does_not_require_sparse_vectors() -> None:
    async def _run() -> tuple[list[str], _FailingSparseEmbedder, TurboPufferFakeNamespace]:
        namespace = TurboPufferFakeNamespace()
        store = _store(namespace)
        await store.upsert(
            [
                VectorPoint(
                    id="dense",
                    dense_vector=[1.0, 0.0, 0.0],
                    sparse_vector=SparseVector(indices=[], values=[]),
                    payload={
                        "namespace": "team-space",
                        "collection": "corpus-a",
                        "document_id": "doc-dense",
                        "content_type": "document",
                        "source_type": "file",
                        "text": "semantic only",
                    },
                ),
                VectorPoint(
                    id="lexical",
                    dense_vector=[0.0, 1.0, 0.0],
                    sparse_vector=SparseVector(indices=[], values=[]),
                    payload={
                        "namespace": "team-space",
                        "collection": "corpus-a",
                        "document_id": "doc-lexical",
                        "content_type": "document",
                        "source_type": "file",
                        "text": "fox fox lexical",
                    },
                ),
            ]
        )
        sparse = _FailingSparseEmbedder()
        results = await HybridRetrieve().retrieve(
            PipelineQuery(
                query="fox",
                namespace="team-space",
                collections=["corpus-a"],
                query_plan=query_plan_preset("hybrid_rrf", limit=2),
            ),
            PipelineContext(
                embedding_provider=FakeEmbeddingProvider(
                    vocabulary=("semantic", "lexical", "fox")
                ),
                sparse_embedder=sparse,
                vector_store=store,
            ),
        )
        document_ids: list[str] = []
        for result in results:
            if result.document_id is None:
                raise AssertionError("expected stored document_id")
            document_ids.append(result.document_id)
        return document_ids, sparse, namespace

    document_ids, sparse, namespace = asyncio.run(_run())

    assert set(document_ids) == {"doc-dense", "doc-lexical"}
    assert sparse.embed_query_multi_calls == []
    rank_by_values = [call["rank_by"] for call in namespace.query_calls]
    assert ("vector", "ANN", [0.0, 0.0, 1.0]) in rank_by_values
    assert (TURBOPUFFER_BM25_TEXT_FIELD, "BM25", "fox") in rank_by_values


def test_turbopuffer_store_bm25_query_plan_uses_lexical_query() -> None:
    async def _run() -> None:
        namespace = TurboPufferFakeNamespace()
        store = _store(namespace)
        await store.upsert(
            [
                VectorPoint(
                    id="alpha",
                    dense_vector=[1.0, 0.0, 0.0],
                    sparse_vector=SparseVector(indices=[1], values=[1.0]),
                    payload={
                        "namespace": "team-space",
                        "collection": "corpus-a",
                        "document_id": "doc-alpha",
                        "content_type": "document",
                        "source_type": "file",
                        "text": "alpha fox",
                    },
                )
            ]
        )

        results = await store.search(
            SearchQuery(
                dense_vector=[0.0, 0.0, 0.0],
                sparse_vector=SparseVector(indices=[1], values=[1.0]),
                namespace="team-space",
                collections=["corpus-a"],
                limit=1,
                lexical_query="fox",
                query_plan=query_plan_preset("sparse_only", limit=1),
            )
        )

        assert namespace.query_calls[-1]["rank_by"] == (
            TURBOPUFFER_BM25_TEXT_FIELD,
            "BM25",
            "fox",
        )
        assert namespace.query_calls[-1]["top_k"] == 20
        assert results[0].document_id == "doc-alpha"
        assert store.capabilities.query_plan.sparse is True
        assert store.capabilities.query_plan.hybrid is True

    asyncio.run(_run())


def test_turbopuffer_store_batches_large_upserts() -> None:
    async def _run() -> None:
        namespace = _RecordingNamespace()
        store = TurboPufferVectorStore(
            namespace="docs",
            dense_dimensions=3,
            namespace_client=namespace,
            write_batch_size=2,
        )

        await store.upsert([_point(f"point-{index}") for index in range(5)])

        batches = [
            cast(list[dict[str, object]], call["upsert_rows"])
            for call in namespace.write_calls
        ]
        assert [[row["id"] for row in rows] for rows in batches] == [
            ["point-0", "point-1"],
            ["point-2", "point-3"],
            ["point-4"],
        ]
        assert all(
            call["distance_metric"] == "cosine_distance"
            for call in namespace.write_calls
        )
        assert namespace.query_calls == []

    asyncio.run(_run())


def test_turbopuffer_store_rejects_wrong_dense_dimensions_before_provider_write() -> (
    None
):
    namespace = _RecordingNamespace()
    store = _store(namespace)
    point = VectorPoint(
        id="point-1",
        dense_vector=[1.0, 0.0],
        sparse_vector=SparseVector(indices=[], values=[]),
        payload={"text": "alpha"},
    )

    with pytest.raises(
        ValueError,
        match="turbopuffer dense vector dimension mismatch at point index 0",
    ):
        asyncio.run(store.upsert([point]))

    assert namespace.write_calls == []


def test_turbopuffer_store_rejects_wrong_query_dimensions_before_provider_search() -> (
    None
):
    namespace = _RecordingNamespace()
    store = _store(namespace)
    query = SearchQuery(
        dense_vector=[1.0, 0.0],
        sparse_vector=SparseVector(indices=[], values=[]),
        namespace="team-space",
        collections=["corpus-a"],
    )

    with pytest.raises(ValueError, match="turbopuffer dense query dimension mismatch"):
        asyncio.run(store.search(query))

    assert namespace.query_calls == []


def test_turbopuffer_store_search_translates_filters_and_results() -> None:
    async def _run() -> None:
        namespace = _RecordingNamespace()
        store = _store(namespace)

        results = await store.search(
            SearchQuery(
                dense_vector=[1.0, 0.0, 0.0],
                sparse_vector=SparseVector(indices=[], values=[]),
                namespace="team-space",
                collections=["corpus-a"],
                limit=5,
                metadata_filter=Range(field="chunk_index", gte=2, lt=8),
            )
        )

        call = namespace.query_calls[0]
        assert call["rank_by"] == ("vector", "ANN", [1.0, 0.0, 0.0])
        assert call["top_k"] == 5
        assert call["include_attributes"] is True
        assert call["filters"] == (
            "And",
            (
                ("namespace", "Eq", "team-space"),
                ("collection", "In", ("corpus-a",)),
                ("And", (("chunk_index", "Gte", 2), ("chunk_index", "Lt", 8))),
            ),
        )
        assert results[0].id == "point-1"
        assert results[0].text == "alpha"
        # TurboPuffer returns cosine distance; the adapter exposes cosine similarity.
        assert results[0].score == pytest.approx(0.75)

    asyncio.run(_run())


def test_turbopuffer_store_empty_allowlist_search_returns_empty_without_query() -> None:
    async def _run() -> None:
        namespace = _RecordingNamespace()
        store = _store(namespace)

        results = await store.search(
            SearchQuery(
                dense_vector=[1.0, 0.0, 0.0],
                sparse_vector=SparseVector(indices=[], values=[]),
                namespace="team-space",
                collections=[],
            )
        )

        assert results == []
        assert namespace.query_calls == []

    asyncio.run(_run())


def test_turbopuffer_store_converts_scores_by_distance_metric() -> None:
    async def _run() -> None:
        cosine_results = await _store_with_metric(
            _RecordingNamespace(),
            distance_metric="cosine_distance",
        ).search(
            SearchQuery(
                dense_vector=[1.0, 0.0, 0.0],
                sparse_vector=SparseVector(indices=[], values=[]),
                namespace="team-space",
                collections=["corpus-a"],
            )
        )
        euclidean_results = await _store_with_metric(
            _RecordingNamespace(),
            distance_metric="euclidean_squared",
        ).search(
            SearchQuery(
                dense_vector=[1.0, 0.0, 0.0],
                sparse_vector=SparseVector(indices=[], values=[]),
                namespace="team-space",
                collections=["corpus-a"],
            )
        )

        assert cosine_results[0].score == pytest.approx(0.75)
        assert euclidean_results[0].score == pytest.approx(0.8)

    asyncio.run(_run())


def test_turbopuffer_store_health_uses_metadata() -> None:
    async def _run() -> None:
        store = _store(_RecordingNamespace())
        health = await store.check_health()

        assert health["healthy"] is True
        assert health["adapter"] == "turbopuffer"
        assert "backend" not in health
        assert health["points_count"] == 3
        assert health["logical_bytes"] == 99
        assert health["index_status"] == "up-to-date"

    asyncio.run(_run())


def test_turbopuffer_store_document_lookup_uses_supported_lookup_and_count_queries() -> (
    None
):
    async def _run() -> None:
        namespace = _RecordingNamespace()
        store = _store(namespace)

        record = await store.get_document_record(
            namespace="team-space",
            collection="corpus-a",
            document_id="doc-1",
        )

        lookup_call = namespace.query_calls[0]
        assert lookup_call["rank_by"] == ("id", "asc")
        assert lookup_call["limit"] == 1
        assert lookup_call["include_attributes"] is True
        assert "aggregate_by" not in lookup_call
        assert lookup_call["filters"] == (
            "And",
            (
                ("namespace", "Eq", "team-space"),
                ("collection", "Eq", "corpus-a"),
                ("document_id", "Eq", "doc-1"),
            ),
        )
        count_call = namespace.query_calls[1]
        assert "rank_by" not in count_call
        assert "include_attributes" not in count_call
        assert count_call["limit"] == 1
        assert count_call["aggregate_by"] == {"chunk_count": ("Count",)}
        assert count_call["filters"] == lookup_call["filters"]
        assert record is not None
        assert record.document_id == "doc-1"
        assert record.document_key == "/doc.md"
        assert record.chunk_count == 7

    asyncio.run(_run())


def test_turbopuffer_store_document_key_lookup_counts_resolved_document_id() -> None:
    class _SharedKeyNamespace(_RecordingNamespace):
        async def query(self, **kwargs: object) -> object:
            self.query_calls.append(kwargs)
            if "aggregate_by" in kwargs:
                return SimpleNamespace(aggregations={"chunk_count": 2})
            return SimpleNamespace(
                rows=[
                    {
                        "id": "point-1",
                        "$dist": 0.25,
                        "namespace": "team-space",
                        "collection": "corpus-a",
                        "document_id": "doc-1",
                        "document_key": "/shared.md",
                        "content_sha256": "sha",
                        "processing_version": "v1",
                        "content_type": "document",
                        "source_type": "file",
                        "text": "alpha",
                        "chunk_index": 0,
                    }
                ]
            )

    async def _run() -> None:
        namespace = _SharedKeyNamespace()
        store = _store(namespace)

        record = await store.get_document_record(
            namespace="team-space",
            collection="corpus-a",
            document_key="/shared.md",
        )

        lookup_call = namespace.query_calls[0]
        assert lookup_call["filters"] == (
            "And",
            (
                ("namespace", "Eq", "team-space"),
                ("collection", "Eq", "corpus-a"),
                ("document_key", "Eq", "/shared.md"),
            ),
        )
        count_call = namespace.query_calls[1]
        assert count_call["filters"] == (
            "And",
            (
                ("namespace", "Eq", "team-space"),
                ("collection", "Eq", "corpus-a"),
                ("document_id", "Eq", "doc-1"),
            ),
        )
        assert record is not None
        assert record.document_id == "doc-1"
        assert record.document_key == "/shared.md"
        assert record.chunk_count == 2

    asyncio.run(_run())


def test_turbopuffer_store_delete_by_filter_allows_partial_continuation() -> None:
    async def _run() -> None:
        namespace = _RecordingNamespace(rows_remaining=[True, False])
        store = _store(namespace)

        await store.delete(
            DeleteFilter(
                namespace="team-space",
                collection="corpus-a",
                document_id="doc-1",
            )
        )

        assert len(namespace.write_calls) == 2
        for call in namespace.write_calls:
            assert call["delete_by_filter_allow_partial"] is True
            assert call["delete_by_filter"] == (
                "And",
                (
                    ("namespace", "Eq", "team-space"),
                    ("collection", "Eq", "corpus-a"),
                    ("document_id", "Eq", "doc-1"),
                ),
            )

    asyncio.run(_run())


def test_turbopuffer_store_delete_by_filter_rejects_invalid_continuation_state() -> (
    None
):
    async def _run() -> None:
        namespace = _RecordingNamespace(rows_remaining=["0"])
        store = _store(namespace)

        with pytest.raises(
            ValueError,
            match="turbopuffer delete response returned invalid rows_remaining",
        ):
            await store.delete(
                DeleteFilter(
                    namespace="team-space",
                    collection="corpus-a",
                    document_id="doc-1",
                )
            )

        assert len(namespace.write_calls) == 1

    asyncio.run(_run())


def test_turbopuffer_store_delete_uses_configured_continuation_limit() -> None:
    async def _run() -> None:
        namespace = _RecordingNamespace(rows_remaining=[True, True])
        store = TurboPufferVectorStore(
            namespace="docs",
            dense_dimensions=3,
            namespace_client=namespace,
            delete_continuation_limit=1,
        )

        with pytest.raises(TurboPufferDeleteByFilterExhausted) as exc_info:
            await store.delete(
                DeleteFilter(
                    namespace="team-space",
                    collection="corpus-a",
                    document_id="doc-1",
                )
            )

        assert exc_info.value.outcome.writes_attempted == 1
        assert len(namespace.write_calls) == 1

    asyncio.run(_run())


def test_turbopuffer_delete_by_filter_exhaustion_raises_with_state() -> None:
    async def _run() -> None:
        namespace = _RecordingNamespace(rows_remaining=[True, True])

        with pytest.raises(
            TurboPufferDeleteByFilterExhausted,
            match="exhausted continuation limit",
        ) as exc_info:
            await delete_turbopuffer_filter(
                namespace_client=namespace,
                filter_values=DeleteFilter(
                    namespace="team-space",
                    collection="corpus-a",
                    document_id="doc-1",
                ),
                policy=DEFAULT_POLICY,
                continuation_limit=2,
            )

        outcome = exc_info.value.outcome
        assert outcome.exhausted is True
        assert outcome.rows_remaining is True
        assert outcome.writes_attempted == 2
        assert len(namespace.write_calls) == 2

    asyncio.run(_run())


def test_turbopuffer_delete_by_filter_exhaustion_can_return_explicit_status() -> None:
    async def _run() -> None:
        namespace = _RecordingNamespace(rows_remaining=[True, True, False])

        outcome = await delete_turbopuffer_filter(
            namespace_client=namespace,
            filter_values=DeleteFilter(
                namespace="team-space",
                collection="corpus-a",
                document_id="doc-1",
            ),
            policy=DEFAULT_POLICY,
            continuation_limit=2,
            raise_on_exhausted=False,
        )

        assert outcome.exhausted is True
        assert outcome.rows_remaining is True
        assert outcome.writes_attempted == 2
        assert len(namespace.write_calls) == 2

    asyncio.run(_run())


def test_turbopuffer_delete_by_filter_rejects_non_positive_continuation_limit() -> None:
    async def _run() -> None:
        namespace = _RecordingNamespace()

        with pytest.raises(
            ValueError,
            match="delete_continuation_limit must be positive",
        ):
            await delete_turbopuffer_filter(
                namespace_client=namespace,
                filter_values=DeleteFilter(
                    namespace="team-space",
                    collection="corpus-a",
                    document_id="doc-1",
                ),
                policy=DEFAULT_POLICY,
                continuation_limit=0,
            )

        assert namespace.write_calls == []

    asyncio.run(_run())


@pytest.mark.live
def test_turbopuffer_live_smoke_when_configured() -> None:
    if not os.environ.get("TURBOPUFFER_API_KEY"):
        pytest.skip("set TURBOPUFFER_API_KEY to run the live smoke")
    pytest.importorskip("turbopuffer")

    async def _run() -> None:
        from turbopuffer import AsyncTurbopuffer

        namespace_name = f"rag-core-smoke-{uuid.uuid4().hex}"
        client = AsyncTurbopuffer(region=os.environ.get("TURBOPUFFER_REGION"))
        namespace = client.namespace(namespace_name)
        store = TurboPufferVectorStore(
            namespace=namespace_name,
            dense_dimensions=3,
            namespace_client=namespace,
        )
        try:
            await store.upsert(
                [
                    VectorPoint(
                        id=str(uuid.uuid4()),
                        dense_vector=[1.0, 0.0, 0.0],
                        sparse_vector=SparseVector(indices=[], values=[]),
                        payload={
                            "namespace": "smoke",
                            "collection": "corpus",
                            "document_id": "doc",
                            "document_key": "/doc.md",
                            "content_sha256": "sha",
                            "processing_version": "v1",
                            "content_type": "document",
                            "source_type": "file",
                            "text": "live smoke",
                            "chunk_index": 0,
                        },
                    )
                ]
            )
            results = await store.search(
                SearchQuery(
                    dense_vector=[1.0, 0.0, 0.0],
                    sparse_vector=SparseVector(indices=[], values=[]),
                    namespace="smoke",
                    collections=["corpus"],
                    limit=1,
                )
            )
            assert results
            assert results[0].document_id == "doc"
            hybrid_results = await store.search(
                SearchQuery(
                    dense_vector=[1.0, 0.0, 0.0],
                    sparse_vector=SparseVector(indices=[1], values=[1.0]),
                    namespace="smoke",
                    collections=["corpus"],
                    limit=1,
                    lexical_query="live smoke",
                    query_plan=query_plan_preset("hybrid_rrf", limit=1),
                )
            )
            assert hybrid_results
            assert hybrid_results[0].document_id == "doc"
        finally:
            await namespace.delete_all()
            await client.close()

    asyncio.run(_run())
