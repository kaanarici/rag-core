from __future__ import annotations
from collections.abc import Sequence


def parse_metadata_fields(values: Sequence[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        key, separator, raw = value.partition("=")
        if not separator or not key.strip():
            raise ValueError("metadata entries must use KEY=VALUE")
        parsed[key.strip()] = raw
    return parsed


def parse_non_empty_values(values: Sequence[str], *, field: str) -> list[str] | None:
    parsed = [value.strip() for value in values]
    if any(not value for value in parsed):
        raise ValueError(f"{field} values must be non-empty")
    return parsed or None
