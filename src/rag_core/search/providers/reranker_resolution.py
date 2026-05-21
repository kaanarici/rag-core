"""Reranker provider resolution and runtime diagnostics."""

from __future__ import annotations

from typing import TypeVar

from rag_core.config.env_access import get_env as config_get_env
from rag_core.search.providers.registry import RERANKER_PROVIDERS

_R = TypeVar("_R", bound=object)
_RERANKER_API_KEY_ENVS = {
    "cohere": ("COHERE_API_KEY", "CO_API_KEY"),
    "voyage": ("VOYAGE_API_KEY",),
    "zeroentropy": ("ZEROENTROPY_API_KEY",),
}


class _SanitizedRerankerInitError(RuntimeError):
    def __init__(self, *, provider: str, error_type: str) -> None:
        self.provider = provider
        self.error_type = error_type
        super().__init__(
            "reranker provider '%s' initialization failed (error_type=%s)"
            % (provider, error_type)
        )


def resolve_reranker_provider(
    provider: str,
    api_key: str | None = None,
) -> tuple[str, str | None]:
    """Resolve a requested reranker provider to an effective provider."""

    requested = (provider or "none").strip().lower()
    if requested == "none":
        return "none", None

    if requested == "cohere":
        key = _resolve_api_key(api_key, env_names=reranker_api_key_env_names("cohere"))
        if key:
            return "cohere", None
        return "none", "missing_cohere_api_key"
    if requested == "voyage":
        key = _resolve_api_key(api_key, env_names=reranker_api_key_env_names("voyage"))
        if key:
            return "voyage", None
        return "none", "missing_voyage_api_key"
    if requested == "zeroentropy":
        key = _resolve_api_key(api_key, env_names=reranker_api_key_env_names("zeroentropy"))
        if key:
            return "zeroentropy", None
        return "none", "missing_zeroentropy_api_key"

    if requested in RERANKER_PROVIDERS:
        return requested, None

    return "invalid", f"unknown_provider:{requested}"


def reranker_api_key_env_names(provider: str) -> tuple[str, ...]:
    return _RERANKER_API_KEY_ENVS.get(provider, ())


def _resolve_api_key(api_key: str | None, *, env_names: tuple[str, ...]) -> str:
    explicit = _normalize_optional_str(api_key)
    if explicit:
        return explicit
    for env_name in env_names:
        key = _normalize_optional_str(config_get_env(env_name))
        if key:
            return key
    return ""


def _normalize_optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return stripped


def attach_runtime_metadata(
    reranker: _R,
    *,
    requested: str,
    effective: str,
    fallback_reason: str | None,
) -> _R:
    from rag_core.core_runtime import (
        RERANKER_EFFECTIVE_ATTR,
        RERANKER_FALLBACK_REASON_ATTR,
        RERANKER_REQUESTED_ATTR,
    )

    setattr(reranker, RERANKER_REQUESTED_ATTR, requested)
    setattr(reranker, RERANKER_EFFECTIVE_ATTR, effective)
    setattr(reranker, RERANKER_FALLBACK_REASON_ATTR, fallback_reason)
    return reranker
