from __future__ import annotations

from rag_core.config.env_access import get_env as config_get_env
from rag_core.provider_api_keys import ANTHROPIC_API_KEY_ENVS, first_configured_api_key
from rag_core.search.providers.registry import CONTEXTUALIZER_PROVIDERS

from .contextualizer import ChunkContextualizer, NoOpContextualizer
from .contextualizer_adapters import AnthropicChunkContextualizer
from .contextualizer_provider_names import (
    ANTHROPIC_CONTEXTUALIZER_ID,
    CONTEXTUALIZER_DISABLED_ALIAS,
    DEFAULT_ANTHROPIC_CONTEXTUALIZER_MODEL,
    NOOP_CONTEXTUALIZER_ID,
)


def create_contextualizer(
    provider: str,
    *,
    model: str | None = None,
    contextualizer_chunk_cap: int | None = None,
) -> ChunkContextualizer:
    requested = _normalize_provider(provider)
    if requested == ANTHROPIC_CONTEXTUALIZER_ID:
        api_key = first_configured_api_key(
            ANTHROPIC_API_KEY_ENVS,
            get_env=config_get_env,
        )
        if not api_key:
            raise ValueError(
                "Anthropic contextualizer requires ANTHROPIC_API_KEY"
            )
        return CONTEXTUALIZER_PROVIDERS.create(
            requested,
            model=model,
            contextualizer_chunk_cap=contextualizer_chunk_cap,
            api_key=api_key,
        )
    return CONTEXTUALIZER_PROVIDERS.create(
        requested,
        model=model,
        contextualizer_chunk_cap=contextualizer_chunk_cap,
    )


def _normalize_provider(provider: str) -> str:
    requested = (provider or "").strip().lower()
    if requested == CONTEXTUALIZER_DISABLED_ALIAS:
        return NOOP_CONTEXTUALIZER_ID
    return requested


def _build_noop_contextualizer(**_: object) -> NoOpContextualizer:
    return NoOpContextualizer()


def _build_anthropic_contextualizer(
    *,
    model: str | None = None,
    contextualizer_chunk_cap: int | None = None,
    api_key: str | None = None,
    **_: object,
) -> AnthropicChunkContextualizer:
    return AnthropicChunkContextualizer(
        model=model or DEFAULT_ANTHROPIC_CONTEXTUALIZER_MODEL,
        chunk_cap=contextualizer_chunk_cap,
        api_key=api_key,
    )


CONTEXTUALIZER_PROVIDERS.register(NOOP_CONTEXTUALIZER_ID, _build_noop_contextualizer)
CONTEXTUALIZER_PROVIDERS.register(
    ANTHROPIC_CONTEXTUALIZER_ID,
    _build_anthropic_contextualizer,
)


__all__ = ["CONTEXTUALIZER_PROVIDERS", "create_contextualizer"]
