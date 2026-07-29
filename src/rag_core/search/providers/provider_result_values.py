"""Sanitized provider result value labels."""

from __future__ import annotations

from collections.abc import Sequence


def safe_provider_value_type(value: object) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if value is None:
        return "none"
    if isinstance(value, list):
        return "list"
    if isinstance(value, tuple):
        return "tuple"
    if isinstance(value, Sequence):
        return "sequence"
    return "object"


__all__ = ["safe_provider_value_type"]
