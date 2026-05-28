from __future__ import annotations

from rag_core.provider_api_keys import (
    PROVIDER_NAMES_WITH_API_KEYS,
    all_provider_api_key_env_names,
    provider_api_key_env_names,
)
from rag_core.provider_package_names import (
    ANTHROPIC_PACKAGE,
    COHERE_PACKAGE,
    GEMINI_PACKAGE,
    GOOGLE_PACKAGE_ALIAS,
    MISTRAL_PACKAGE,
    OPENAI_PACKAGE,
    PROVIDER_ERROR_MODULES,
    VOYAGE_PACKAGE,
    ZEROENTROPY_PACKAGE,
)

_BOOTSTRAP_TERMS = (
    "api key",
    "api_key",
    "apikey",
    "auth",
    "credential",
    "missing key",
    "unauthorized",
)


class ProviderCliError(ValueError):
    pass


def is_provider_bootstrap_error(exc: Exception) -> bool:
    if isinstance(exc, ProviderCliError):
        return False
    module = type(exc).__module__.split(".", 1)[0]
    message = str(exc).lower()
    if module in PROVIDER_ERROR_MODULES:
        return any(term in message for term in _BOOTSTRAP_TERMS)
    return "api key" in message and any(
        provider in message for provider in PROVIDER_NAMES_WITH_API_KEYS
    )


def is_provider_error(exc: Exception) -> bool:
    return type(exc).__module__.split(".", 1)[0] in PROVIDER_ERROR_MODULES


def provider_bootstrap_message(exc: Exception, *, action: str) -> str:
    provider = _provider_name(exc)
    error_type = type(exc).__name__
    env_vars = _provider_env_vars(provider)
    return (
        f"provider setup failed before {action}: "
        f"provider={provider} error_type={error_type}. "
        "Run `rag-core doctor --json` and check the persistent-search "
        f"configuration, including {provider} provider API key env vars: "
        f"{', '.join(env_vars)}."
    )


def provider_runtime_message(exc: Exception, *, action: str) -> str:
    provider = _provider_name(exc)
    error_type = type(exc).__name__
    return (
        f"provider failed during {action}: "
        f"provider={provider} error_type={error_type}. "
        "Run `rag-core doctor --json` to inspect provider and runtime "
        "configuration. Provider exception details were hidden; retry with "
        "application-owned logging if you need provider-specific diagnostics."
    )


def _provider_name(exc: Exception) -> str:
    module = type(exc).__module__.split(".", 1)[0]
    if module == VOYAGE_PACKAGE:
        return "voyage"
    if module == GOOGLE_PACKAGE_ALIAS:
        return "gemini"
    if module in {
        ANTHROPIC_PACKAGE,
        OPENAI_PACKAGE,
        COHERE_PACKAGE,
        GEMINI_PACKAGE,
        MISTRAL_PACKAGE,
        ZEROENTROPY_PACKAGE,
    }:
        return module
    message = str(exc).lower()
    for provider in PROVIDER_NAMES_WITH_API_KEYS:
        if provider in message:
            if provider == VOYAGE_PACKAGE:
                return "voyage"
            if provider == GOOGLE_PACKAGE_ALIAS:
                return "gemini"
            return provider
    return "unknown"


def _provider_env_vars(provider: str) -> tuple[str, ...]:
    return provider_api_key_env_names(provider) or all_provider_api_key_env_names()
