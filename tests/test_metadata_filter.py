"""Coverage for the typed metadata-filter AST.

The AST is the engine's vendor-neutral way to express filters beyond the
first-class ``namespace``/``corpus_ids``/``document_ids``/``content_types``
fields. These tests pin down construction validation, the in-memory
interpreter's payload semantics, the Qdrant translator's ``rest.Filter`` shape,
and round-trip behaviour through the search pipeline.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, cast

import pytest
from qdrant_client import models as rest

from rag_core.search.filter_eval import eval_filter
from rag_core.search.providers.memory_store import InMemoryVectorStore
from rag_core.search.providers.qdrant_filters import metadata_filter_to_qdrant
from rag_core.search.query_plan import UnsupportedQueryStage
from rag_core.search.pipeline_runner import (
    SearchExecutionOptions,
    SearchPipelineRunner,
    SearchRequest,
)
from rag_core.search.filters import (
    And,
    Filter,
    Geo,
    In,
    Not,
    Or,
    Range,
    Term,
)
from rag_core.search.request_models import SearchQuery
from rag_core.search.vector_models import (
    SparseVector,
    VectorPoint,
)

from tests.support import FakeEmbeddingProvider, FakeSparseEmbedder


# ---------------------------------------------------------------------------
# Construction validation


def test_range_rejects_fully_open_bounds() -> None:
    with pytest.raises(ValueError, match="Range requires"):
        Range(field="published_at")


def test_in_rejects_empty_values() -> None:
    with pytest.raises(ValueError, match="In.values"):
        In(field="language", values=())


@pytest.mark.parametrize("ast_cls", [And, Or])
def test_boolean_filters_reject_empty_inputs(ast_cls: type) -> None:
    with pytest.raises(ValueError, match=f"{ast_cls.__name__}.filters"):
        ast_cls(filters=())


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"lat": 120.0, "lon": 0.0, "radius_m": 10.0}, "Geo.lat"),
        ({"lat": 0.0, "lon": 200.0, "radius_m": 10.0}, "Geo.lon"),
        ({"lat": 0.0, "lon": 0.0, "radius_m": 0.0}, "Geo.radius_m"),
    ],
)
def test_geo_rejects_invalid_coordinates(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Geo(field="loc", **kwargs)


# ---------------------------------------------------------------------------
# In-memory filter interpreter


def test_eval_filter_term_in_range_combinations() -> None:
    payload = {"language": "en", "category": "news", "published_at": 1700.0}
    assert eval_filter(Term(field="language", value="en"), payload) is True
    assert eval_filter(Term(field="language", value="fr"), payload) is False
    assert eval_filter(In(field="category", values=("news", "blog")), payload) is True
    assert eval_filter(In(field="category", values=("blog",)), payload) is False
    assert eval_filter(Range(field="published_at", gte=1000.0, lte=2000.0), payload) is True
    assert eval_filter(Range(field="published_at", gt=1700.0), payload) is False


def test_eval_filter_range_supports_same_type_strings() -> None:
    payload = {"published_at": "2026-05-20T10:00:00Z"}
    assert (
        eval_filter(
            Range(
                field="published_at",
                gte=cast(Any, "2026-05-01"),
                lt=cast(Any, "2026-06-01"),
            ),
            payload,
        )
        is True
    )
    assert (
        eval_filter(
            Range(field="published_at", lt=cast(Any, "2026-05-01")),
            payload,
        )
        is False
    )


def test_eval_filter_range_rejects_mixed_value_and_bound_types() -> None:
    assert (
        eval_filter(
            Range(field="published_at", gte=cast(Any, "2026-05-01")),
            {"published_at": 1700.0},
        )
        is False
    )
    assert (
        eval_filter(
            Range(field="published_at", gte=1000.0),
            {"published_at": "2026-05-20"},
        )
        is False
    )


def test_eval_filter_returns_false_when_payload_field_missing() -> None:
    assert eval_filter(Term(field="absent", value="x"), {}) is False
    assert eval_filter(Range(field="absent", gte=0.0), {}) is False
    assert eval_filter(In(field="absent", values=("x",)), {}) is False


def test_eval_filter_geo_uses_haversine_radius() -> None:
    # Empire State Building -> Times Square is roughly 1.3 km apart.
    payload = {"loc": {"lat": 40.7484, "lon": -73.9857}}
    near = Geo(field="loc", lat=40.758, lon=-73.9855, radius_m=2_000.0)
    far = Geo(field="loc", lat=40.758, lon=-73.9855, radius_m=500.0)
    assert eval_filter(near, payload) is True
    assert eval_filter(far, payload) is False


def test_eval_filter_boolean_composition() -> None:
    payload = {"language": "en", "category": "news", "published_at": 1500.0}
    composite = And(
        filters=(
            In(field="category", values=("news", "blog")),
            Or(filters=(Term(field="language", value="en"), Term(field="language", value="fr"))),
            Not(filter=Range(field="published_at", lt=1000.0)),
        )
    )
    assert eval_filter(composite, payload) is True

    bad = And(
        filters=(
            Term(field="language", value="es"),
            Range(field="published_at", gte=0.0),
        )
    )
    assert eval_filter(bad, payload) is False


# ---------------------------------------------------------------------------
# Filter applies to real searches


def test_memory_store_metadata_filter_narrows_candidates() -> None:
    async def _run() -> None:
        store = InMemoryVectorStore()
        await store.upsert(
            [
                _payload_point("p1", language="en", category="news", published_at=1000.0),
                _payload_point("p2", language="en", category="news", published_at=2000.0),
                _payload_point("p3", language="fr", category="news", published_at=1500.0),
                _payload_point("p4", language="en", category="blog", published_at=1500.0),
                _payload_point("p5", language="en", category="news", published_at=500.0),
            ]
        )

        metadata_filter: Filter = And(
            filters=(
                In(field="language", values=("en",)),
                In(field="category", values=("news",)),
                Range(field="published_at", gte=1000.0, lte=2000.0),
            )
        )
        results = await store.search(
            SearchQuery(
                dense_vector=[1.0, 0.0, 0.0],
                sparse_vector=SparseVector(indices=[], values=[]),
                namespace="ns",
                corpus_ids=["corpus"],
                limit=10,
                metadata_filter=metadata_filter,
            )
        )
        assert sorted(hit.id for hit in results) == ["p1", "p2"]

    asyncio.run(_run())


def test_search_pipeline_runner_round_trips_metadata_filter() -> None:
    async def _run() -> None:
        store = InMemoryVectorStore()
        await store.upsert(
            [
                _payload_point("p1", language="en", category="news", published_at=1500.0),
                _payload_point("p2", language="en", category="blog", published_at=1500.0),
                _payload_point("p3", language="fr", category="news", published_at=1500.0),
            ]
        )

        pipeline_runner = SearchPipelineRunner(
            embedding_provider=FakeEmbeddingProvider(vocabulary=("a",)),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=store,
        )

        metadata_filter: Filter = And(
            filters=(
                Term(field="language", value="en"),
                Term(field="category", value="news"),
            )
        )
        results = await pipeline_runner.search(
            SearchRequest(
                query="anything",
                corpus_ids=["corpus"],
                namespace="ns",
                metadata_filter=metadata_filter,
                execution=SearchExecutionOptions(
                    query_vector=[1.0, 0.0, 0.0],
                    query_sparse_vectors={
                        "bm25": SparseVector(indices=[1], values=[1.0])
                    },
                ),
            )
        )
        assert [hit.id for hit in results] == ["p1"]

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Qdrant translator


def test_metadata_filter_to_qdrant_translates_nested_shape() -> None:
    metadata_filter: Filter = And(
        filters=(
            In(field="language", values=("en", "fr")),
            Range(field="published_at", gte=1000.0, lte=2000.0),
            Not(filter=Term(field="category", value="spam")),
            Or(
                filters=(
                    Geo(field="loc", lat=10.0, lon=20.0, radius_m=5_000.0),
                    Term(field="featured", value=True),
                )
            ),
        )
    )

    translated = metadata_filter_to_qdrant(metadata_filter)
    expected = rest.Filter(
        must=[
            rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="language", match=rest.MatchAny(any=["en", "fr"])
                    ),
                    rest.FieldCondition(
                        key="published_at",
                        range=rest.Range(gte=1000.0, lte=2000.0, gt=None, lt=None),
                    ),
                    rest.Filter(
                        must_not=[
                            rest.FieldCondition(
                                key="category", match=rest.MatchValue(value="spam")
                            )
                        ]
                    ),
                    rest.Filter(
                        should=[
                            rest.FieldCondition(
                                key="loc",
                                geo_radius=rest.GeoRadius(
                                    center=rest.GeoPoint(lat=10.0, lon=20.0),
                                    radius=5_000.0,
                                ),
                            ),
                            rest.FieldCondition(
                                key="featured", match=rest.MatchValue(value=True)
                            ),
                        ]
                    ),
                ]
            )
        ]
    )
    assert translated == expected


def test_metadata_filter_to_qdrant_rejects_string_ranges() -> None:
    with pytest.raises(UnsupportedQueryStage, match="string Range"):
        metadata_filter_to_qdrant(
            Range(field="published_at", gte="2026-05-01", lt="2026-06-01")
        )


def test_metadata_filter_to_qdrant_raises_on_unknown_node() -> None:
    @dataclass(frozen=True)
    class Mystery:
        field: str = "x"

    with pytest.raises(UnsupportedQueryStage, match="Mystery"):
        metadata_filter_to_qdrant(Mystery())  # type: ignore[arg-type]


def _payload_point(
    point_id: str,
    *,
    language: str,
    category: str,
    published_at: float,
) -> VectorPoint:
    return VectorPoint(
        id=point_id,
        dense_vector=[1.0, 0.0, 0.0],
        sparse_vector=SparseVector(indices=[1], values=[1.0]),
        payload={
            "namespace": "ns",
            "corpus_id": "corpus",
            "document_id": point_id,
            "content_type": "document",
            "source_type": "file",
            "text": point_id,
            "chunk_index": 0,
            "language": language,
            "category": category,
            "published_at": published_at,
        },
    )
