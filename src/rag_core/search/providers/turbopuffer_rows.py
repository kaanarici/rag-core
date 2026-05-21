"""TurboPuffer response row coercion helpers."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Protocol, cast

from rag_core.search.policy import VectorStorePolicy
from rag_core.search.stored_payload import payload_to_result
from rag_core.search.types import SearchResult

_MISSING = object()


class _ToDict(Protocol):
    def to_dict(self) -> Mapping[str, object]: ...


def _row_to_result(
    row: object,
    *,
    distance_metric: str,
    policy: VectorStorePolicy,
) -> SearchResult:
    return payload_to_result(
        point_id=_required_row_id(row),
        payload=_row_payload(row),
        score=_distance_to_score(
            _required_row_float(row, "$dist"),
            distance_metric=distance_metric,
        ),
        policy=policy,
    )


def _required_response_rows(response: object, *, operation: str) -> list[object]:
    rows = getattr(response, "rows", _MISSING)
    if rows is _MISSING or rows is None:
        raise ValueError("turbopuffer %s response missing required rows" % operation)
    try:
        return list(cast(Iterable[object], rows))
    except TypeError:
        raise ValueError("turbopuffer %s response returned invalid rows" % operation) from None


def _row_payload(row: object) -> dict[str, object]:
    if isinstance(row, Mapping):
        raw = dict(row)
    elif hasattr(row, "model_extra"):
        raw = dict(getattr(row, "model_extra") or {})
        raw["id"] = getattr(row, "id")
    elif hasattr(row, "to_dict"):
        raw = dict(cast(_ToDict, row).to_dict())
    else:
        raw = dict(getattr(row, "__dict__", {}))

    return {
        str(key): value
        for key, value in raw.items()
        if key not in {"id", "vector", "$dist"}
    }


def _required_row_value(row: object, key: str) -> object:
    value = _row_value(row, key, default=_MISSING)
    if value is _MISSING:
        raise ValueError("turbopuffer result row missing required field: %s" % key)
    return value


def _required_row_float(row: object, key: str) -> float:
    value = _required_row_value(row, key)
    parsed = _optional_float(value)
    if parsed is None:
        raise ValueError("turbopuffer result row returned invalid field: %s" % key)
    if key == "$dist" and parsed < 0.0:
        raise ValueError("turbopuffer result row returned invalid field: %s" % key)
    return parsed


def _required_row_id(row: object) -> str:
    value = _required_row_value(row, "id")
    return _non_empty_string(
        value,
        "turbopuffer result row missing required field: id",
    )


def _non_empty_string(value: object, message: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(message)
    return value


def _row_value(row: object, key: str, *, default: object = _MISSING) -> object:
    if isinstance(row, Mapping):
        return row.get(key, default)
    try:
        return row[key]  # type: ignore[index]
    except (IndexError, KeyError, TypeError):
        return getattr(row, key, default)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        parsed = float(value)
    except (OverflowError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _distance_to_score(
    distance: float,
    *,
    distance_metric: str,
) -> float:
    if distance_metric == "cosine_distance":
        return 1.0 - min(distance, 2.0)
    return 1.0 / (1.0 + distance)
