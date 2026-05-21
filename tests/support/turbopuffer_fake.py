"""In-memory TurboPuffer namespace for shared vector-store contract tests."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import cast


class TurboPufferFakeNamespace:
    """Minimal namespace that stores upsert rows and answers ANN queries."""

    def __init__(self) -> None:
        self.write_calls: list[dict[str, object]] = []
        self.query_calls: list[dict[str, object]] = []
        self._rows: dict[str, dict[str, object]] = {}

    async def metadata(self) -> object:
        return SimpleNamespace(
            approx_row_count=len(self._rows),
            approx_logical_bytes=0,
            index=SimpleNamespace(status="up-to-date"),
        )

    async def write(self, **kwargs: object) -> object:
        self.write_calls.append(kwargs)
        upsert_rows = kwargs.get("upsert_rows")
        if isinstance(upsert_rows, list):
            for row in upsert_rows:
                if isinstance(row, dict) and row.get("id") is not None:
                    self._rows[str(row["id"])] = dict(row)
        deletes = kwargs.get("deletes")
        if isinstance(deletes, list):
            for point_id in deletes:
                self._rows.pop(str(point_id), None)
        delete_filter = kwargs.get("delete_by_filter")
        if delete_filter is not None:
            remaining = {
                point_id: row
                for point_id, row in self._rows.items()
                if not _matches_filters(row, delete_filter)
            }
            self._rows = remaining
        return SimpleNamespace(rows_remaining=False)

    async def query(self, **kwargs: object) -> object:
        self.query_calls.append(kwargs)
        if "queries" in kwargs:
            subqueries = kwargs["queries"]
            if isinstance(subqueries, list):
                return SimpleNamespace(
                    results=[await self.query(**subquery) for subquery in subqueries]
                )
        filters = kwargs.get("filters")
        if kwargs.get("aggregate_by") is not None:
            matched = [
                row
                for row in self._rows.values()
                if _matches_filters(row, filters)
            ]
            return SimpleNamespace(
                rows=matched[:1],
                aggregations={"chunk_count": len(matched)},
            )
        rank_by = kwargs.get("rank_by")
        top_k = _int_kwarg(kwargs.get("top_k") or kwargs.get("limit") or 10)
        candidates = [
            row
            for row in self._rows.values()
            if _matches_filters(row, filters)
        ]
        if isinstance(rank_by, tuple) and len(rank_by) == 2 and rank_by[1] == "asc":
            field = str(rank_by[0])
            candidates.sort(key=lambda row: str(row.get(field) or ""))
            return SimpleNamespace(rows=candidates[:top_k])
        if isinstance(rank_by, tuple) and len(rank_by) == 3:
            field, mode, vector = rank_by
            if field == "vector" and mode == "ANN" and isinstance(vector, list):
                scored = [
                    (row, _cosine_distance(vector, cast(list[float], row.get("vector") or [])))
                    for row in candidates
                ]
                scored.sort(key=lambda item: item[1])
                rows = [
                    {**row, "$dist": dist}
                    for row, dist in scored[:top_k]
                ]
                return SimpleNamespace(rows=rows)
            if field == "sparse_vector" and mode == "SparseKNN" and isinstance(vector, dict):
                scored = [
                    (
                        row,
                        _sparse_distance(
                            cast(dict[str, float], vector),
                            cast(dict[str, float], row.get("sparse_vector") or {}),
                        ),
                    )
                    for row in candidates
                    if row.get("sparse_vector")
                ]
                scored.sort(key=lambda item: item[1])
                rows = [
                    {**row, "$dist": dist}
                    for row, dist in scored[:top_k]
                ]
                return SimpleNamespace(rows=rows)
        return SimpleNamespace(rows=[])

    async def delete_all(self) -> None:
        self._rows.clear()


def _matches_filters(row: dict[str, object], filters: object) -> bool:
    if filters is None:
        return True
    if isinstance(filters, tuple) and len(filters) == 2:
        op, children = filters
        if op == "And":
            return all(_matches_filters(row, child) for child in cast(tuple[object], children))
        if op == "Or":
            return any(_matches_filters(row, child) for child in cast(tuple[object], children))
        if op == "Not":
            return not _matches_filters(row, children)
    if isinstance(filters, tuple) and len(filters) == 3:
        field, op, expected = filters
        value = row.get(str(field))
        if op == "Eq":
            return bool(value == expected)
        if op == "In":
            return value in cast(tuple[object], expected)
        if op == "Gte":
            return value is not None and value >= expected
        if op == "Gt":
            return value is not None and value > expected
        if op == "Lte":
            return value is not None and value <= expected
        if op == "Lt":
            return value is not None and value < expected
    return True


def _int_kwarg(value: object) -> int:
    if isinstance(value, bool):
        return 10
    if isinstance(value, (int, str)):
        return int(value)
    return 10


def _cosine_distance(query: list[float], candidate: list[float]) -> float:
    if len(query) != len(candidate) or not query:
        return 1.0
    dot = sum(left * right for left, right in zip(query, candidate, strict=True))
    left_norm = math.sqrt(sum(value * value for value in query))
    right_norm = math.sqrt(sum(value * value for value in candidate))
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    similarity = dot / (left_norm * right_norm)
    return 1.0 - similarity


def _sparse_distance(query: dict[str, float], candidate: dict[str, float]) -> float:
    keys = set(query) | set(candidate)
    if not keys:
        return 1.0
    dot = sum(query.get(key, 0.0) * candidate.get(key, 0.0) for key in keys)
    left_norm = math.sqrt(sum(value * value for value in query.values()))
    right_norm = math.sqrt(sum(value * value for value in candidate.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    return 1.0 - (dot / (left_norm * right_norm))
