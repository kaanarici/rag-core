from __future__ import annotations

from typing import Any, cast

import pytest

from rag_core.search.types import SearchQuery, SparseVector, VectorPoint


def test_vector_point_accepts_finite_dense_vector_values() -> None:
    np = pytest.importorskip("numpy")

    point = VectorPoint(
        id="point-1",
        dense_vector=[cast(Any, np.float32(0.5)), cast(Any, np.float64(1.5))],
        sparse_vector=SparseVector(indices=[], values=[]),
        payload={"text": "alpha"},
    )

    assert point.dense_vector[0] == pytest.approx(0.5)


def test_search_query_accepts_sparse_only_empty_dense_vector() -> None:
    query = SearchQuery(
        dense_vector=[],
        sparse_vector=SparseVector(indices=[1], values=[1.0]),
        namespace="team-space",
        corpus_ids=["corpus-a"],
    )

    assert query.dense_vector == []


@pytest.mark.parametrize(
    "value",
    [True, float("nan"), float("inf"), -float("inf"), cast(Any, "1")],
)
def test_vector_point_rejects_malformed_dense_vector_values(value: object) -> None:
    with pytest.raises(ValueError) as exc_info:
        VectorPoint(
            id="point-1",
            dense_vector=[cast(Any, value)],
            sparse_vector=SparseVector(indices=[], values=[]),
            payload={"text": "alpha"},
        )

    assert str(exc_info.value) == "VectorPoint.dense_vector must contain finite numbers"


@pytest.mark.parametrize(
    "value",
    [True, float("nan"), float("inf"), -float("inf"), cast(Any, "1")],
)
def test_search_query_rejects_malformed_dense_vector_values(value: object) -> None:
    with pytest.raises(ValueError) as exc_info:
        SearchQuery(
            dense_vector=[cast(Any, value)],
            sparse_vector=SparseVector(indices=[], values=[]),
            namespace="team-space",
            corpus_ids=["corpus-a"],
        )

    assert str(exc_info.value) == "SearchQuery.dense_vector must contain finite numbers"
