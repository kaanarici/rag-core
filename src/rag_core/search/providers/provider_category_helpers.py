"""Shared helpers for provider category diagnostics."""

from __future__ import annotations

import importlib.util

from rag_core.provider_api_keys import api_key_configured as provider_api_key_configured

from .diagnostic_support import (
    FIELD_CONFIGURED,
    FIELD_PACKAGE_AVAILABLE,
    FIELD_RUNTIME_CONFIG,
    FIELD_SUPPORT_LEVEL,
    SUPPORT_INJECTED,
)


def add_injected_provider(
    providers: dict[str, object],
    configured: str | None,
    *,
    known: tuple[str, ...],
) -> None:
    if configured is None or configured in known:
        return
    providers[configured] = {
        FIELD_SUPPORT_LEVEL: SUPPORT_INJECTED,
        FIELD_CONFIGURED: True,
        FIELD_PACKAGE_AVAILABLE: None,
        FIELD_RUNTIME_CONFIG: "direct constructor injection",
    }


def normalize_runtime_provider(
    value: str | None,
    *,
    default: str | None = None,
) -> str | None:
    normalized = normalize(value)
    if not normalized:
        return default
    return normalized


def api_env_configured(
    env_names: tuple[str, ...],
    *,
    explicit_key: str | None = None,
) -> bool:
    return provider_api_key_configured(
        env_names,
        explicit_key=explicit_key,
    )


def package_available(
    provider: str,
    *,
    packages_by_provider: dict[str, str],
) -> bool:
    try:
        return importlib.util.find_spec(packages_by_provider[provider]) is not None
    except ModuleNotFoundError:
        return False


def normalize(value: str | None) -> str:
    return (value or "").strip().lower()


__all__ = [
    "add_injected_provider",
    "api_env_configured",
    "normalize",
    "normalize_runtime_provider",
    "package_available",
]
