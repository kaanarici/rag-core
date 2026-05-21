"""TurboPuffer write operation helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rag_core.search.policy import VectorStorePolicy
from rag_core.search.types import DeleteFilter, VectorPoint

from .turbopuffer_client import TurboPufferNamespace
from .turbopuffer_config import validate_turbopuffer_write_batch_size
from .turbopuffer_filters import _delete_filter
from .turbopuffer_payloads import _point_to_row, _schema, _validate_point_id
from .vector_dimensions import validate_point_dense_dimensions

DEFAULT_TURBOPUFFER_DELETE_CONTINUATION_LIMIT = 1000


@dataclass(frozen=True)
class TurboPufferDeleteByFilterOutcome:
    writes_attempted: int
    exhausted: bool
    rows_remaining: bool


class TurboPufferDeleteByFilterExhausted(ValueError):
    def __init__(self, *, outcome: TurboPufferDeleteByFilterOutcome) -> None:
        self.outcome = outcome
        super().__init__(
            "turbopuffer delete by filter exhausted continuation limit "
            f"(writes_attempted={outcome.writes_attempted}, rows_remaining={outcome.rows_remaining})"
        )


async def upsert_turbopuffer_points(
    *,
    namespace_client: TurboPufferNamespace,
    points: Sequence[VectorPoint],
    dense_dimensions: int,
    distance_metric: str,
    write_batch_size: int,
    policy: VectorStorePolicy,
) -> None:
    if not points:
        return
    write_batch_size = validate_turbopuffer_write_batch_size(write_batch_size)
    validate_point_dense_dimensions(
        points,
        dense_dimensions=dense_dimensions,
        backend="turbopuffer",
    )
    rows = [_point_to_row(point) for point in points]
    schema = _schema(dense_dimensions, policy=policy)
    for index in range(0, len(rows), write_batch_size):
        await namespace_client.write(
            upsert_rows=rows[index : index + write_batch_size],
            schema=schema,
            distance_metric=distance_metric,
        )


async def delete_turbopuffer_filter(
    *,
    namespace_client: TurboPufferNamespace,
    filter_values: DeleteFilter,
    policy: VectorStorePolicy,
    continuation_limit: int = DEFAULT_TURBOPUFFER_DELETE_CONTINUATION_LIMIT,
    raise_on_exhausted: bool = True,
) -> TurboPufferDeleteByFilterOutcome:
    namespace = (filter_values.namespace or "").strip()
    if not namespace:
        raise ValueError("namespace is required for delete")
    continuation_limit = _validate_delete_continuation_limit(continuation_limit)
    delete_filter = _delete_filter(
        filter_values=filter_values,
        namespace=namespace,
        policy=policy,
    )
    for writes_attempted in range(1, continuation_limit + 1):
        response = await namespace_client.write(
            delete_by_filter=delete_filter,
            delete_by_filter_allow_partial=True,
        )
        rows_remaining = getattr(response, "rows_remaining", False)
        if not isinstance(rows_remaining, bool):
            raise ValueError(
                "turbopuffer delete response returned invalid rows_remaining"
            )
        if not rows_remaining:
            return TurboPufferDeleteByFilterOutcome(
                writes_attempted=writes_attempted,
                exhausted=False,
                rows_remaining=False,
            )
    exhausted = TurboPufferDeleteByFilterOutcome(
        writes_attempted=continuation_limit,
        exhausted=True,
        rows_remaining=True,
    )
    if raise_on_exhausted:
        raise TurboPufferDeleteByFilterExhausted(outcome=exhausted)
    return exhausted


async def delete_turbopuffer_point_ids(
    *,
    namespace_client: TurboPufferNamespace,
    point_ids: Sequence[str],
) -> None:
    if not point_ids:
        return
    await namespace_client.write(
        deletes=[_validate_point_id(point_id) for point_id in point_ids]
    )


def _validate_delete_continuation_limit(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("turbopuffer delete continuation_limit must be positive")
    return value
