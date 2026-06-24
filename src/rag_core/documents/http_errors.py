from __future__ import annotations


def safe_http_status(exc: object) -> int | str:
    code = getattr(exc, "code", None)
    if isinstance(code, bool) or not isinstance(code, int):
        return "unknown"
    return code


__all__ = ["safe_http_status"]
