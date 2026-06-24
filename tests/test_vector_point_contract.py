from __future__ import annotations

from typing import Any, cast

import pytest

from rag_core.search.vector_models import (
    SparseVector,
    VectorPoint,
)


def _point(point_id: object) -> VectorPoint:
    return VectorPoint(
        id=cast(Any, point_id),
        dense_vector=[1.0],
        sparse_vector=SparseVector(indices=[], values=[]),
        payload={"text": "alpha"},
    )


def test_vector_point_accepts_non_empty_string_id() -> None:
    point = _point("point-1")

    assert point.id == "point-1"


@pytest.mark.parametrize(
    "point_id",
    [
        pytest.param("", id="empty"),
        pytest.param("   ", id="blank"),
        pytest.param(True, id="bool"),
        pytest.param(123, id="int"),
        pytest.param(object(), id="object"),
    ],
)
def test_vector_point_rejects_invalid_ids(point_id: object) -> None:
    with pytest.raises(ValueError) as exc_info:
        _point(point_id)

    assert str(exc_info.value) == "VectorPoint.id must be a non-empty string"
