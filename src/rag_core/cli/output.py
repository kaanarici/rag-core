from __future__ import annotations

from dataclasses import asdict, is_dataclass


def search_hit_payload(hit: object) -> dict[str, object]:
    if is_dataclass(hit) and not isinstance(hit, type):
        return {key: value for key, value in asdict(hit).items() if value is not None}
    return require_mapping(hit)


def float_value(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def require_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}
