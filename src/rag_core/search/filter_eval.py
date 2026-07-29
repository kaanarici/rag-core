from __future__ import annotations

import math
from collections.abc import Mapping

from rag_core.search.filters import And, Filter, Geo, In, Not, Or, Range, Term


def eval_filter(filter: Filter, payload: Mapping[str, object]) -> bool:
    """Walk a ``Filter`` AST against a payload mapping."""
    if isinstance(filter, Term):
        value = payload.get(filter.field)
        if value is None:
            return False
        return value == filter.value
    if isinstance(filter, In):
        value = payload.get(filter.field)
        if value is None:
            return False
        return value in filter.values
    if isinstance(filter, Range):
        value = payload.get(filter.field)
        if value is None:
            return False
        return _eval_range(value, filter)
    if isinstance(filter, Geo):
        point = payload.get(filter.field)
        if not isinstance(point, Mapping):
            return False
        try:
            lat = float(point["lat"])
            lon = float(point["lon"])
        except (KeyError, TypeError, ValueError):
            return False
        return _haversine_m(lat, lon, filter.lat, filter.lon) <= filter.radius_m
    if isinstance(filter, And):
        return all(eval_filter(child, payload) for child in filter.filters)
    if isinstance(filter, Or):
        return any(eval_filter(child, payload) for child in filter.filters)
    if isinstance(filter, Not):
        return not eval_filter(filter.filter, payload)
    raise TypeError(f"unknown Filter node: {type(filter).__name__}")


def _eval_range(value: object, filter: Range) -> bool:
    bounds = (filter.gte, filter.gt, filter.lte, filter.lt)
    present_bounds = tuple(bound for bound in bounds if bound is not None)
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        if any(
            isinstance(bound, bool) or not isinstance(bound, (int, float))
            for bound in present_bounds
        ):
            return False
        return _eval_numeric_range(
            float(value),
            gte=_float_bound(filter.gte),
            gt=_float_bound(filter.gt),
            lte=_float_bound(filter.lte),
            lt=_float_bound(filter.lt),
        )
    if isinstance(value, str):
        if not all(isinstance(bound, str) for bound in present_bounds):
            return False
        return _eval_string_range(
            value,
            gte=filter.gte if isinstance(filter.gte, str) else None,
            gt=filter.gt if isinstance(filter.gt, str) else None,
            lte=filter.lte if isinstance(filter.lte, str) else None,
            lt=filter.lt if isinstance(filter.lt, str) else None,
        )
    return False


def _eval_numeric_range(
    value: float,
    *,
    gte: float | None,
    gt: float | None,
    lte: float | None,
    lt: float | None,
) -> bool:
    if gte is not None and not value >= gte:
        return False
    if gt is not None and not value > gt:
        return False
    if lte is not None and not value <= lte:
        return False
    if lt is not None and not value < lt:
        return False
    return True


def _eval_string_range(
    value: str,
    *,
    gte: str | None,
    gt: str | None,
    lte: str | None,
    lt: str | None,
) -> bool:
    if gte is not None and not value >= gte:
        return False
    if gt is not None and not value > gt:
        return False
    if lte is not None and not value <= lte:
        return False
    if lt is not None and not value < lt:
        return False
    return True


def _float_bound(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_m = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(delta_lambda / 2.0) ** 2
    )
    return earth_radius_m * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
