"""Small environment helpers used by the parsers and scoped facade."""

from __future__ import annotations
import os


def get_env_optional(name: str) -> str | None:
    return os.getenv(name)


def get_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default
