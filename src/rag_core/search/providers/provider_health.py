from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from rag_core.search.provider_protocols import ProviderHealth

ProviderHealthKind = str

PROVIDER_HEALTH_KIND_EMBEDDING: Final[ProviderHealthKind] = "embedding"
PROVIDER_HEALTH_KIND_RERANKER: Final[ProviderHealthKind] = "reranker"
PROVIDER_HEALTH_ERROR_AUTH: Final[str] = "auth"
PROVIDER_HEALTH_ERROR_NETWORK: Final[str] = "network"
PROVIDER_HEALTH_ERROR_PROVIDER: Final[str] = "provider"

_AUTH_STATUS_CODES = frozenset({401, 403})


def build_healthy_provider_health(
    *,
    provider_name: str,
    kind: ProviderHealthKind,
    model_name: str | None = None,
    dimensions: int | None = None,
    requested_provider: str | None = None,
    fallback_reason: str | None = None,
) -> ProviderHealth:
    payload: ProviderHealth = {
        "healthy": True,
        "adapter": provider_name,
        "kind": kind,
    }
    if model_name is not None:
        payload["model"] = model_name
    if dimensions is not None:
        payload["dimensions"] = dimensions
    if requested_provider is not None:
        payload["requested"] = requested_provider
    if fallback_reason is not None:
        payload["fallback_reason"] = fallback_reason
    return payload


def build_unhealthy_provider_health(
    *,
    provider_name: str,
    kind: ProviderHealthKind,
    model_name: str | None,
    dimensions: int | None = None,
    exc: Exception,
    transient: bool,
    api_key_envs: Sequence[str] | None = None,
) -> ProviderHealth:
    category = provider_health_error_category(exc, transient=transient)
    payload: ProviderHealth = {
        "healthy": False,
        "adapter": provider_name,
        "kind": kind,
        "error": type(exc).__name__,
        "error_category": category,
        "message": provider_health_message(
            provider_name=provider_name,
            kind=kind,
            category=category,
            model_name=model_name,
            api_key_envs=(
                provider_api_key_env_names(provider_name)
                if api_key_envs is None
                else api_key_envs
            ),
        ),
    }
    if model_name is not None:
        payload["model"] = model_name
    if dimensions is not None:
        payload["dimensions"] = dimensions
    return payload


def provider_health_error_category(exc: Exception, *, transient: bool) -> str:
    status = _status_code(exc)
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    if (
        status in _AUTH_STATUS_CODES
        or "auth" in name
        or "permission" in name
        or "api key" in message
        or "api_key" in message
        or "credential" in message
        or "unauthorized" in message
    ):
        return PROVIDER_HEALTH_ERROR_AUTH
    if transient or "connection" in name or "timeout" in name or "network" in name:
        return PROVIDER_HEALTH_ERROR_NETWORK
    return PROVIDER_HEALTH_ERROR_PROVIDER


def provider_health_message(
    *,
    provider_name: str,
    kind: ProviderHealthKind,
    category: str,
    model_name: str | None,
    api_key_envs: Sequence[str] = (),
) -> str:
    subject = f"{provider_name} {kind} health check"
    model_hint = f" for model {model_name}" if model_name else ""
    if category == PROVIDER_HEALTH_ERROR_AUTH:
        env_hint = _env_hint(api_key_envs)
        return f"{subject} failed authentication{model_hint}; verify {env_hint}."
    if category == PROVIDER_HEALTH_ERROR_NETWORK:
        return (
            f"{subject} could not reach the provider{model_hint}; "
            "check network access and provider availability."
        )
    return (
        f"{subject} failed{model_hint}; verify the provider package, model, "
        "dimensions, and account access."
    )


def _status_code(exc: Exception) -> int | None:
    for attr in ("status_code", "http_status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _env_hint(api_key_envs: Sequence[str]) -> str:
    names = [name for name in api_key_envs if name]
    if not names:
        return "the provider API key configuration"
    return " or ".join(names)


def provider_api_key_env_names(provider_name: str) -> tuple[str, ...]:
    from rag_core.provider_api_keys import provider_api_key_env_names as _env_names

    return _env_names(provider_name)


__all__ = [
    "PROVIDER_HEALTH_ERROR_AUTH",
    "PROVIDER_HEALTH_ERROR_NETWORK",
    "PROVIDER_HEALTH_ERROR_PROVIDER",
    "PROVIDER_HEALTH_KIND_EMBEDDING",
    "PROVIDER_HEALTH_KIND_RERANKER",
    "build_healthy_provider_health",
    "build_unhealthy_provider_health",
    "provider_api_key_env_names",
    "provider_health_error_category",
    "provider_health_message",
]
