from __future__ import annotations

import os
from collections.abc import Callable

from rag_core.config import DEFAULT_EMBEDDING_PROVIDER
from rag_core.provider_package_names import VOYAGE_PACKAGE
from rag_core.search.providers.cohere import COHERE_PROVIDER
from rag_core.search.providers.voyage import VOYAGE_PROVIDER
from rag_core.search.providers.zeroentropy import ZEROENTROPY_PROVIDER

ANTHROPIC_API_PROVIDER = "anthropic"
GEMINI_API_PROVIDER = "gemini"
GOOGLE_API_PROVIDER_ALIAS = "google"
MISTRAL_API_PROVIDER = "mistral"
VOYAGE_API_PACKAGE_ALIAS = VOYAGE_PACKAGE
ANTHROPIC_API_KEY_ENVS = ("ANTHROPIC_API_KEY",)
COHERE_API_KEY_ENVS = ("COHERE_API_KEY", "CO_API_KEY")
GEMINI_API_KEY_ENVS = ("GOOGLE_API_KEY", "GEMINI_API_KEY")
GOOGLE_API_KEY_ENVS = GEMINI_API_KEY_ENVS
MISTRAL_API_KEY_ENVS = ("MISTRAL_API_KEY",)
OPENAI_API_KEY_ENVS = ("OPENAI_API_KEY",)
QDRANT_API_KEY_ENVS = ("RAG_CORE_QDRANT_API_KEY",)
TURBOPUFFER_API_KEY_ENVS = ("TURBOPUFFER_API_KEY",)
VOYAGE_API_KEY_ENVS = ("VOYAGE_API_KEY",)
ZEROENTROPY_API_KEY_ENVS = ("ZEROENTROPY_API_KEY",)

PROVIDER_API_KEY_ENVS = {
    ANTHROPIC_API_PROVIDER: ANTHROPIC_API_KEY_ENVS,
    DEFAULT_EMBEDDING_PROVIDER: OPENAI_API_KEY_ENVS,
    COHERE_PROVIDER: COHERE_API_KEY_ENVS,
    GEMINI_API_PROVIDER: GEMINI_API_KEY_ENVS,
    GOOGLE_API_PROVIDER_ALIAS: GOOGLE_API_KEY_ENVS,
    MISTRAL_API_PROVIDER: MISTRAL_API_KEY_ENVS,
    VOYAGE_PROVIDER: VOYAGE_API_KEY_ENVS,
    VOYAGE_API_PACKAGE_ALIAS: VOYAGE_API_KEY_ENVS,
    ZEROENTROPY_PROVIDER: ZEROENTROPY_API_KEY_ENVS,
}
PROVIDER_NAMES_WITH_API_KEYS = tuple(PROVIDER_API_KEY_ENVS)


def provider_api_key_env_names(provider: str) -> tuple[str, ...]:
    return PROVIDER_API_KEY_ENVS.get(provider, ())


def all_provider_api_key_env_names() -> tuple[str, ...]:
    env_names: list[str] = []
    for provider_env_names in PROVIDER_API_KEY_ENVS.values():
        for env_name in provider_env_names:
            if env_name not in env_names:
                env_names.append(env_name)
    return tuple(env_names)


def normalize_api_key(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip()


def first_configured_api_key(
    env_names: tuple[str, ...],
    *,
    explicit_key: str | None = None,
    get_env: Callable[[str], str | None] = os.environ.get,
) -> str:
    explicit = normalize_api_key(explicit_key)
    if explicit:
        return explicit
    for env_name in env_names:
        key = normalize_api_key(get_env(env_name))
        if key:
            return key
    return ""


def api_key_configured(
    env_names: tuple[str, ...],
    *,
    explicit_key: str | None = None,
    get_env: Callable[[str], str | None] = os.environ.get,
) -> bool:
    return bool(
        first_configured_api_key(
            env_names,
            explicit_key=explicit_key,
            get_env=get_env,
        )
    )


__all__ = [
    "ANTHROPIC_API_KEY_ENVS",
    "ANTHROPIC_API_PROVIDER",
    "COHERE_API_KEY_ENVS",
    "GEMINI_API_KEY_ENVS",
    "GEMINI_API_PROVIDER",
    "GOOGLE_API_PROVIDER_ALIAS",
    "GOOGLE_API_KEY_ENVS",
    "MISTRAL_API_KEY_ENVS",
    "MISTRAL_API_PROVIDER",
    "OPENAI_API_KEY_ENVS",
    "PROVIDER_API_KEY_ENVS",
    "PROVIDER_NAMES_WITH_API_KEYS",
    "QDRANT_API_KEY_ENVS",
    "TURBOPUFFER_API_KEY_ENVS",
    "VOYAGE_API_KEY_ENVS",
    "VOYAGE_API_PACKAGE_ALIAS",
    "ZEROENTROPY_API_KEY_ENVS",
    "all_provider_api_key_env_names",
    "api_key_configured",
    "first_configured_api_key",
    "normalize_api_key",
    "provider_api_key_env_names",
]
