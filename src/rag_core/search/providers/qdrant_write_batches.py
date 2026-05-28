from __future__ import annotations

from collections.abc import Sequence

from qdrant_client import models as rest

from rag_core.search.vector_models import VectorPoint

from .qdrant_payloads import _build_point


def build_qdrant_point_batches(
    *,
    points: Sequence[VectorPoint],
    batch_size: int,
    available_sparse_vector_names: frozenset[str] | set[str],
) -> list[list[rest.PointStruct]]:
    qdrant_points = [
        _build_point(
            point,
            available_sparse_vector_names=available_sparse_vector_names,
        )
        for point in points
    ]
    return split_into_batches(qdrant_points, batch_size)


def split_into_batches(
    points: list[rest.PointStruct],
    batch_size: int,
) -> list[list[rest.PointStruct]]:
    if batch_size <= 0:
        return [points] if points else []
    return [points[i : i + batch_size] for i in range(0, len(points), batch_size)]


__all__ = ["build_qdrant_point_batches", "split_into_batches"]
