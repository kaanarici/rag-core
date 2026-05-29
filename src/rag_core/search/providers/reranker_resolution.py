"""Reranker provider resolution and runtime diagnostics."""

from __future__ import annotations

from typing import TypeVar

from rag_core.config import DEFAULT_RERANKER_PROVIDER
from rag_core.config.env_access import get_env as config_get_env
from rag_core.provider_api_keys import (
    COHERE_API_KEY_ENVS,
    VOYAGE_API_KEY_ENVS,
    ZEROENTROPY_API_KEY_ENVS,
    first_configured_api_key,
)
from rag_core.search.providers.cohere import COHERE_RERANKER_PROVIDER
from rag_core.search.providers.registry import RERANKER_PROVIDERS
from rag_core.search.providers.voyage import VOYAGE_PROVIDER
from rag_core.search.providers.zeroentropy import ZEROENTROPY_PROVIDER

_R = TypeVar("_R", bound=object)
_RERANKER_API_KEY_ENVS = {
    COHERE_RERANKER_PROVIDER: COHERE_API_KEY_ENVS,
    VOYAGE_PROVIDER: VOYAGE_API_KEY_ENVS,
    ZEROENTROPY_PROVIDER: ZEROENTROPY_API_KEY_ENVS,
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

    requested = (provider or DEFAULT_RERANKER_PROVIDER).strip().lower()
    if requested == DEFAULT_RERANKER_PROVIDER:
        return DEFAULT_RERANKER_PROVIDER, None

    if requested == COHERE_RERANKER_PROVIDER:
        key = _resolve_api_key(
            api_key,
            env_names=reranker_api_key_env_names(COHERE_RERANKER_PROVIDER),
        )
        if key:
            return COHERE_RERANKER_PROVIDER, None
        return DEFAULT_RERANKER_PROVIDER, "missing_cohere_api_key"
    if requested == VOYAGE_PROVIDER:
        key = _resolve_api_key(
            api_key,
            env_names=reranker_api_key_env_names(VOYAGE_PROVIDER),
        )
        if key:
            return VOYAGE_PROVIDER, None
        return DEFAULT_RERANKER_PROVIDER, "missing_voyage_api_key"
    if requested == ZEROENTROPY_PROVIDER:
        key = _resolve_api_key(
            api_key,
            env_names=reranker_api_key_env_names(ZEROENTROPY_PROVIDER),
        )
        if key:
            return ZEROENTROPY_PROVIDER, None
        return DEFAULT_RERANKER_PROVIDER, "missing_zeroentropy_api_key"

    if requested in RERANKER_PROVIDERS:
        return requested, None

    return "invalid", f"unknown_provider:{requested}"


def reranker_api_key_env_names(provider: str) -> tuple[str, ...]:
    return _RERANKER_API_KEY_ENVS.get(provider, ())


def _resolve_api_key(api_key: str | None, *, env_names: tuple[str, ...]) -> str:
    return first_configured_api_key(
        env_names,
        explicit_key=api_key,
        get_env=config_get_env,
    )


def attach_runtime_metadata(
    reranker: _R,
    *,
    requested: str,
    effective: str,
    fallback_reason: str | None,
) -> _R:
    from rag_core._engine.core_runtime import (
        RERANKER_EFFECTIVE_ATTR,
        RERANKER_FALLBACK_REASON_ATTR,
        RERANKER_REQUESTED_ATTR,
    )

    setattr(reranker, RERANKER_REQUESTED_ATTR, requested)
    setattr(reranker, RERANKER_EFFECTIVE_ATTR, effective)
    setattr(reranker, RERANKER_FALLBACK_REASON_ATTR, fallback_reason)
    return reranker
