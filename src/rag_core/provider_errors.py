from __future__ import annotations
from rag_core.provider_api_keys import provider_api_key_env_names


class ProviderCliError(ValueError):
    pass


def is_provider_error(exc: Exception) -> bool:
    return type(exc).__module__.split(".", 1)[0] in {"openai", "qdrant_client"}


def is_provider_bootstrap_error(exc: Exception) -> bool:
    return is_provider_error(exc) and any(
        t in str(exc).lower()
        for t in ("api key", "api_key", "credential", "unauthorized", "auth")
    )


def provider_bootstrap_message(exc: Exception, *, action: str) -> str:
    provider = type(exc).__module__.split(".", 1)[0]
    return f"provider setup failed before {action}: provider={provider} error_type={type(exc).__name__}. Check credentials: {', '.join(provider_api_key_env_names(provider))}"


def provider_runtime_message(exc: Exception, *, action: str) -> str:
    return f"provider failed during {action}: error_type={type(exc).__name__}. Provider details hidden; inspect application-owned logs."
