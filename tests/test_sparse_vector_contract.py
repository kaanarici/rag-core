from __future__ import annotations

from typing import Any, cast

import pytest

from rag_core.search.vector_models import SparseVector


def test_sparse_vector_accepts_empty_and_zero_weight_vectors() -> None:
    assert SparseVector(indices=[], values=[]).indices == []

    vector = SparseVector(indices=[0, 2], values=[0.0, 1.5])

    assert vector.values == [0.0, 1.5]


def test_sparse_vector_accepts_provider_shaped_numpy_scalars() -> None:
    np = pytest.importorskip("numpy")

    vector = SparseVector(
        indices=[cast(Any, np.int64(1)), cast(Any, np.uint32(2))],
        values=[cast(Any, np.float32(0.5)), cast(Any, np.float64(1.5))],
    )

    assert vector.indices[0] == 1
    assert float(vector.values[0]) == pytest.approx(0.5)


def test_sparse_vector_rejects_mismatched_index_and_value_lengths() -> None:
    with pytest.raises(ValueError) as exc_info:
        SparseVector(indices=[1], values=[])

    assert str(exc_info.value) == (
        "SparseVector.indices and values must have the same length"
    )


@pytest.mark.parametrize("index", [-1, True, cast(Any, 1.5), cast(Any, "1")])
def test_sparse_vector_rejects_invalid_indices(index: object) -> None:
    with pytest.raises(ValueError) as exc_info:
        SparseVector(indices=[cast(Any, index)], values=[1.0])

    assert str(exc_info.value) == "SparseVector.indices must be non-negative integers"


@pytest.mark.parametrize(
    "value",
    [True, float("nan"), float("inf"), -float("inf"), cast(Any, "1")],
)
def test_sparse_vector_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError) as exc_info:
        SparseVector(indices=[1], values=[cast(Any, value)])

    assert str(exc_info.value) == "SparseVector.values must be finite numbers"
