from __future__ import annotations
from collections.abc import Callable, Sequence
import os

OPENAI_API_KEY_ENVS = ("OPENAI_API_KEY",)
QDRANT_API_KEY_ENVS = ("RAG_CORE_QDRANT_API_KEY", "QDRANT_API_KEY")
PROVIDER_NAMES_WITH_API_KEYS = ("openai", "qdrant")


def first_configured_api_key(
    env_names: Sequence[str],
    *,
    explicit_key: str | None = None,
    get_env: Callable[[str], str | None] = os.getenv,
) -> str:
    for value in (explicit_key, *(get_env(name) for name in env_names)):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def provider_api_key_env_names(provider: str) -> tuple[str, ...]:
    return {"openai": OPENAI_API_KEY_ENVS, "qdrant": QDRANT_API_KEY_ENVS}.get(
        provider, ()
    )


def all_provider_api_key_env_names() -> tuple[str, ...]:
    return OPENAI_API_KEY_ENVS + QDRANT_API_KEY_ENVS


def api_key_configured(
    env_names: Sequence[str], explicit_key: str | None = None
) -> bool:
    return bool(first_configured_api_key(env_names, explicit_key=explicit_key))
