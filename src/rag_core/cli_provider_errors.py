from __future__ import annotations


_PROVIDER_ENV_VARS = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "cohere": ("COHERE_API_KEY", "CO_API_KEY"),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "mistral": ("MISTRAL_API_KEY",),
    "voyage": ("VOYAGE_API_KEY",),
    "voyageai": ("VOYAGE_API_KEY",),
    "turbopuffer": ("TURBOPUFFER_API_KEY",),
    "zeroentropy": ("ZEROENTROPY_API_KEY",),
}
_PROVIDER_NAMES = tuple(_PROVIDER_ENV_VARS)
_PROVIDER_MODULES = {
    "anthropic",
    "cohere",
    "gemini",
    "google",
    "mistral",
    "openai",
    "turbopuffer",
    "voyageai",
    "zeroentropy",
}
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
    if module in _PROVIDER_MODULES:
        return any(term in message for term in _BOOTSTRAP_TERMS)
    return "api key" in message and any(
        provider in message for provider in _PROVIDER_NAMES
    )


def is_provider_error(exc: Exception) -> bool:
    return type(exc).__module__.split(".", 1)[0] in _PROVIDER_MODULES


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
    if module == "voyageai":
        return "voyage"
    if module == "google":
        return "gemini"
    if module in {"anthropic", "openai", "cohere", "gemini", "mistral", "turbopuffer", "zeroentropy"}:
        return module
    message = str(exc).lower()
    for provider in _PROVIDER_NAMES:
        if provider in message:
            if provider == "voyageai":
                return "voyage"
            if provider == "google":
                return "gemini"
            return provider
    return "unknown"


def _provider_env_vars(provider: str) -> tuple[str, ...]:
    if provider in _PROVIDER_ENV_VARS:
        return _PROVIDER_ENV_VARS[provider]
    env_vars: list[str] = []
    for provider_env_vars in _PROVIDER_ENV_VARS.values():
        for env_var in provider_env_vars:
            if env_var not in env_vars:
                env_vars.append(env_var)
    return tuple(env_vars)
