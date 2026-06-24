from __future__ import annotations

import inspect
from collections.abc import Awaitable
from typing import cast

from rag_core.core_models import Config
from rag_core.provider_api_keys import provider_api_key_env_names
from rag_core.search.provider_protocols import ProviderHealth
from rag_core.search.providers.embedding import create_embedding_provider
from rag_core.search.providers.embedding_models import resolve_embedding_dimensions
from rag_core.search.providers.provider_health import (
    PROVIDER_HEALTH_KIND_EMBEDDING,
    PROVIDER_HEALTH_KIND_RERANKER,
    build_unhealthy_provider_health,
)
from rag_core.search.providers.reranker import create_reranker
from rag_core.search.providers.reranker_runtime import (
    RERANKER_EFFECTIVE_ATTR,
    RERANKER_FALLBACK_REASON_ATTR,
    RERANKER_REQUESTED_ATTR,
)


async def exercise_doctor_model_providers(
    config: Config,
) -> dict[str, ProviderHealth]:
    health: dict[str, ProviderHealth] = {}
    embedding_health = await _embedding_provider_health(config)
    if embedding_health is not None:
        health["embedding"] = embedding_health
    reranker_health = await _reranker_provider_health(config)
    if reranker_health is not None:
        health["reranker"] = reranker_health
    return health


async def _embedding_provider_health(config: Config) -> ProviderHealth | None:
    dimensions = resolve_embedding_dimensions(
        provider=config.embedding.provider,
        model=config.embedding.model,
        dimensions=config.embedding.dimensions,
    )
    try:
        provider = create_embedding_provider(
            provider=config.embedding.provider,
            model=config.embedding.model,
            dimensions=config.embedding.dimensions,
            api_key=config.embedding.api_key,
            base_url=config.embedding.base_url,
        )
    except Exception as exc:
        return build_unhealthy_provider_health(
            provider_name=config.embedding.provider,
            kind=PROVIDER_HEALTH_KIND_EMBEDDING,
            model_name=config.embedding.model,
            dimensions=dimensions,
            exc=exc,
            transient=False,
            api_key_envs=provider_api_key_env_names(config.embedding.provider),
        )
    return await _call_optional_check_health(provider)


async def _reranker_provider_health(config: Config) -> ProviderHealth | None:
    try:
        provider = create_reranker(
            provider=config.reranker.provider,
            model=config.reranker.model,
            api_key=config.reranker.api_key,
        )
    except Exception as exc:
        return build_unhealthy_provider_health(
            provider_name=config.reranker.provider,
            kind=PROVIDER_HEALTH_KIND_RERANKER,
            model_name=config.reranker.model,
            exc=exc,
            transient=False,
            api_key_envs=provider_api_key_env_names(config.reranker.provider),
        )
    health = await _call_optional_check_health(provider)
    if health is None:
        return None
    requested = getattr(provider, RERANKER_REQUESTED_ATTR, config.reranker.provider)
    effective = getattr(provider, RERANKER_EFFECTIVE_ATTR, None)
    fallback_reason = getattr(provider, RERANKER_FALLBACK_REASON_ATTR, None)
    if isinstance(requested, str):
        health["requested"] = requested
    if isinstance(effective, str):
        health["effective"] = effective
    if isinstance(fallback_reason, str):
        health["fallback_reason"] = fallback_reason
    return health


async def _call_optional_check_health(provider: object) -> ProviderHealth | None:
    check_health = getattr(provider, "check_health", None)
    if not callable(check_health):
        return None
    result = check_health()
    if not inspect.isawaitable(result):
        return None
    health = await cast(Awaitable[ProviderHealth], result)
    return dict(health)


__all__ = ["exercise_doctor_model_providers"]
