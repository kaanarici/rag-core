"""Sanitized exception type names for document parsing diagnostics."""

from __future__ import annotations


def exception_type(exc: Exception) -> str:
    return type(exc).__name__


def root_exception_type(exc: Exception) -> str:
    cause = exc.__cause__
    if isinstance(cause, Exception):
        return exception_type(cause)
    return exception_type(exc)


__all__ = ["exception_type", "root_exception_type"]
