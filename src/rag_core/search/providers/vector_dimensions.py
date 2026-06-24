from __future__ import annotations

from collections.abc import Sequence

from rag_core.search.vector_models import VectorPoint


def validate_point_dense_dimensions(
    points: Sequence[VectorPoint],
    *,
    dense_dimensions: int,
    provider_name: str,
) -> None:
    if dense_dimensions <= 0:
        raise ValueError("%s dense_dimensions must be positive" % provider_name)
    for index, point in enumerate(points):
        actual_dimensions = len(point.dense_vector)
        if actual_dimensions != dense_dimensions:
            raise ValueError(
                "%s dense vector dimension mismatch at point index %d: "
                "expected %d dimensions, got %d"
                % (provider_name, index, dense_dimensions, actual_dimensions)
            )


def validate_query_dense_dimensions(
    dense_vector: Sequence[float],
    *,
    dense_dimensions: int,
    provider_name: str,
) -> None:
    if not dense_vector:
        return
    if dense_dimensions <= 0:
        raise ValueError("%s dense_dimensions must be positive" % provider_name)
    actual_dimensions = len(dense_vector)
    if actual_dimensions != dense_dimensions:
        raise ValueError(
            "%s dense query dimension mismatch: expected %d dimensions, got %d"
            % (provider_name, dense_dimensions, actual_dimensions)
        )
