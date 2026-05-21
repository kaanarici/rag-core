"""Provider readiness diagnostics used by doctor surfaces."""

from __future__ import annotations

import importlib.util

from rag_core.config.env_access import get_env_stripped
from rag_core.core_models import RAGCoreConfig

from .embedding_models import get_embedding_model_spec
from .openai_embedding import normalize_optional_provider_config_value
from .provider_category_diagnostics import (
    describe_chunk_context_cache_provider_diagnostics,
    describe_contextualizer_diagnostics,
    describe_embedding_cache_provider_diagnostics,
    describe_event_sink_provider_diagnostics,
    describe_ocr_provider_diagnostics,
    describe_search_sidecar_provider_diagnostics,
    describe_sparse_provider_diagnostics,
)
from .registry import EMBEDDING_PROVIDERS, RERANKER_PROVIDERS
from .reranker_resolution import reranker_api_key_env_names, resolve_reranker_provider

_EMBEDDING_PROVIDER_ORDER = ("openai", "voyage", "zeroentropy")
_RERANKER_PROVIDER_ORDER = ("none", "cohere", "voyage", "zeroentropy")
_PACKAGE_BY_PROVIDER = {
    "openai": "openai",
    "cohere": "cohere",
    "voyage": "voyageai",
    "zeroentropy": "zeroentropy",
}
_API_KEY_ENV_BY_PROVIDER = {
    "openai": "OPENAI_API_KEY",
    "cohere": "COHERE_API_KEY",
    "voyage": "VOYAGE_API_KEY",
    "zeroentropy": "ZEROENTROPY_API_KEY",
}


def describe_model_provider_diagnostics(
    *,
    config: RAGCoreConfig,
    embedding_dimensions: int,
    sparse_provider_name: str | None = None,
    ocr_provider_name: str | None = None,
    contextualizer_name: str | None = None,
    embedding_cache_name: str | None = None,
    chunk_context_cache_name: str | None = None,
    search_sidecar_name: str | None = None,
    event_sink_name: str | None = None,
) -> dict[str, object]:
    """Return non-secret provider readiness diagnostics."""

    return {
        "embedding": describe_embedding_provider_diagnostics(
            config=config,
            embedding_dimensions=embedding_dimensions,
        ),
        "sparse": describe_sparse_provider_diagnostics(
            runtime_provider=sparse_provider_name,
        ),
        "reranker": describe_reranker_provider_diagnostics(config=config),
        "ocr": describe_ocr_provider_diagnostics(runtime_provider=ocr_provider_name),
        "contextualizer": describe_contextualizer_diagnostics(
            runtime_provider=contextualizer_name,
        ),
        "embedding_cache": describe_embedding_cache_provider_diagnostics(
            config=config,
            runtime_provider=embedding_cache_name,
        ),
        "chunk_context_cache": describe_chunk_context_cache_provider_diagnostics(
            runtime_provider=chunk_context_cache_name,
        ),
        "search_sidecar": describe_search_sidecar_provider_diagnostics(
            config=config,
            runtime_provider=search_sidecar_name,
        ),
        "event_sink": describe_event_sink_provider_diagnostics(
            runtime_provider=event_sink_name,
        ),
    }


def describe_embedding_provider_diagnostics(
    *,
    config: RAGCoreConfig,
    embedding_dimensions: int,
) -> dict[str, object]:
    configured = _normalize(config.embedding.provider) or "openai"
    return {
        "configured": configured,
        "registered": list(EMBEDDING_PROVIDERS.names()),
        "providers": {
            provider: _embedding_provider_diagnostics(
                provider,
                config=config,
                configured=configured,
                embedding_dimensions=embedding_dimensions,
            )
            for provider in _EMBEDDING_PROVIDER_ORDER
        },
    }


def describe_reranker_provider_diagnostics(
    *,
    config: RAGCoreConfig,
) -> dict[str, object]:
    configured = _normalize(config.reranker.provider) or "none"
    effective, fallback_reason = resolve_reranker_provider(
        configured,
        api_key=config.reranker.api_key,
    )
    return {
        "configured": configured,
        "effective": effective,
        "fallback_reason": fallback_reason,
        "registered": list(RERANKER_PROVIDERS.names()),
        "providers": {
            provider: _reranker_provider_diagnostics(
                provider,
                config=config,
                configured=configured,
            )
            for provider in _RERANKER_PROVIDER_ORDER
        },
    }


def _embedding_provider_diagnostics(
    provider: str,
    *,
    config: RAGCoreConfig,
    configured: str,
    embedding_dimensions: int,
) -> dict[str, object]:
    selected = provider == configured
    spec = (
        get_embedding_model_spec(provider, config.embedding.model) if selected else None
    )
    payload: dict[str, object] = {
        "support_level": "default" if provider == "openai" else "first_party_optional",
        "configured": selected,
        "package_available": _package_available(provider),
        "api_key_env": _API_KEY_ENV_BY_PROVIDER[provider],
        "api_key_configured": _api_key_configured(
            provider,
            configured=configured,
            explicit_key=config.embedding.api_key,
        ),
        "runtime_config": "RAGCoreConfig.embedding",
    }
    if selected:
        payload.update(
            {
                "model": config.embedding.model,
                "dimensions": embedding_dimensions,
                "model_known": spec is not None,
                "dimensions_override": config.embedding.dimensions is not None,
                "batch_size": config.embedding.batch_size,
                "base_url_configured": bool(
                    normalize_optional_provider_config_value(config.embedding.base_url)
                ),
            }
        )
    if spec is not None:
        payload.update(
            {
                "default_dimensions": spec.default_dimensions,
                "max_dimensions": spec.max_dimensions,
                "allowed_dimensions": list(spec.allowed_dimensions or ()),
                "supports_dimensions_override": spec.supports_dimensions_override,
            }
        )
    return payload


def _reranker_provider_diagnostics(
    provider: str,
    *,
    config: RAGCoreConfig,
    configured: str,
) -> dict[str, object]:
    selected = provider == configured
    payload: dict[str, object] = {
        "support_level": "default_noop" if provider == "none" else "first_party_optional",
        "configured": selected,
        "package_available": True if provider == "none" else _package_available(provider),
        "runtime_config": "RAGCoreConfig.reranker",
    }
    if provider != "none":
        payload.update(
            {
                "api_key_env": _api_key_env_payload(provider),
                "api_key_configured": _api_key_configured(
                    provider,
                    configured=configured,
                    explicit_key=config.reranker.api_key,
                ),
            }
        )
    if selected:
        payload["model"] = config.reranker.model
    return payload


def _api_key_configured(
    provider: str,
    *,
    configured: str,
    explicit_key: str | None,
) -> bool:
    selected_explicit_key = explicit_key if provider == configured else None
    return _api_env_configured(
        _api_key_env_names(provider),
        explicit_key=selected_explicit_key,
    )


def _api_key_env_names(provider: str) -> tuple[str, ...]:
    return reranker_api_key_env_names(provider) or (_API_KEY_ENV_BY_PROVIDER[provider],)


def _api_key_env_payload(provider: str) -> str | list[str]:
    env_names = _api_key_env_names(provider)
    if len(env_names) == 1:
        return env_names[0]
    return list(env_names)


def _package_available(provider: str) -> bool:
    return importlib.util.find_spec(_PACKAGE_BY_PROVIDER[provider]) is not None


def _api_env_configured(
    env_names: tuple[str, ...],
    *,
    explicit_key: str | None = None,
) -> bool:
    return bool((explicit_key or "").strip()) or any(
        bool(get_env_stripped(env_name)) for env_name in env_names
    )


def _normalize(value: str | None) -> str:
    return (value or "").strip().lower()


__all__ = [
    "describe_embedding_provider_diagnostics",
    "describe_embedding_cache_provider_diagnostics",
    "describe_chunk_context_cache_provider_diagnostics",
    "describe_contextualizer_diagnostics",
    "describe_event_sink_provider_diagnostics",
    "describe_model_provider_diagnostics",
    "describe_ocr_provider_diagnostics",
    "describe_reranker_provider_diagnostics",
    "describe_search_sidecar_provider_diagnostics",
    "describe_sparse_provider_diagnostics",
]
