from __future__ import annotations

import re
from collections.abc import Sequence

from rag_core.cli_provider_errors import (
    ProviderCliError,
    is_provider_bootstrap_error,
    is_provider_error,
    provider_bootstrap_message,
    provider_runtime_message,
)
from rag_core.fetch_security import redact_fetch_url

_SENSITIVE_ERROR_PATTERNS = (
    re.compile(r"\bapi[_ -]?key\b", re.IGNORECASE),
    re.compile(r"\bsecret\b", re.IGNORECASE),
    re.compile(r"\btoken=", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_-]+"),
)


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


def cli_error_message(exc: Exception) -> str:
    if isinstance(exc, FileNotFoundError):
        return f"file not found: {exc.filename or exc}"
    return str(exc)


def cli_safe_error_message(exc: Exception, *, action: str) -> str:
    if isinstance(exc, ProviderCliError):
        return str(exc)
    if is_provider_bootstrap_error(exc):
        return provider_bootstrap_message(exc, action=action)
    if is_provider_error(exc):
        return provider_runtime_message(exc, action=action)
    message = cli_error_message(exc)
    if _looks_sensitive_error_message(message):
        return f"{action} failed with {type(exc).__name__}"
    return message


def cli_redacted_url(value: str | None) -> str | None:
    if value is None:
        return None
    redacted = redact_fetch_url(value)
    return "<url-configured>" if redacted == "<invalid-url>" else redacted


def cli_store_location_label(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if stripped == ":memory:":
        return stripped
    if "://" in stripped:
        return cli_redacted_url(stripped)
    return "local_path_configured"


def _looks_sensitive_error_message(message: str) -> bool:
    return any(pattern.search(message) for pattern in _SENSITIVE_ERROR_PATTERNS)


__all__ = [
    "cli_error_message",
    "cli_redacted_url",
    "cli_safe_error_message",
    "cli_store_location_label",
    "parse_metadata_fields",
    "parse_non_empty_values",
]
