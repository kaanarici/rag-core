"""Provider readiness diagnostics used by doctor surfaces."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

from rag_core.config import (
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_RERANKER_PROVIDER,
    DEMO_EMBEDDING_PROVIDER,
)
from rag_core.core_models import RAGCoreConfig
from rag_core.provider_api_keys import (
    OPENAI_API_KEY_ENVS,
    VOYAGE_API_KEY_ENVS,
    ZEROENTROPY_API_KEY_ENVS,
    api_key_configured,
)
from rag_core.provider_package_names import (
    COHERE_PACKAGE,
    OPENAI_PACKAGE,
    VOYAGE_PACKAGE,
    ZEROENTROPY_PACKAGE,
)

from .cohere import COHERE_RERANKER_PROVIDER
from .diagnostic_support import (
    FIELD_API_KEY_CONFIGURED,
    FIELD_API_KEY_ENV,
    FIELD_CONFIGURED,
    FIELD_PACKAGE_AVAILABLE,
    FIELD_PROVIDERS,
    FIELD_REGISTERED,
    FIELD_RUNTIME_CONFIG,
    FIELD_SUPPORT_LEVEL,
    SUPPORT_DEFAULT,
    SUPPORT_DEFAULT_NOOP,
    SUPPORT_FIRST_PARTY_OPTIONAL,
    SUPPORT_FIRST_PARTY_UTILITY,
    ProviderDiagnosticSupportLevel,
)
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
from .provider_category_names import (
    CHUNK_CONTEXT_CACHE_PROVIDER_CATEGORY,
    CONTEXTUALIZER_PROVIDER_CATEGORY,
    EMBEDDING_CACHE_PROVIDER_CATEGORY,
    EMBEDDING_PROVIDER_CATEGORY,
    EVENT_SINK_PROVIDER_CATEGORY,
    OCR_PROVIDER_CATEGORY,
    RERANKER_PROVIDER_CATEGORY,
    SEARCH_SIDECAR_PROVIDER_CATEGORY,
    SPARSE_PROVIDER_CATEGORY,
)
from .registry import EMBEDDING_PROVIDERS, RERANKER_PROVIDERS
from .reranker_resolution import reranker_api_key_env_names, resolve_reranker_provider
from .voyage import VOYAGE_PROVIDER
from .zeroentropy import ZEROENTROPY_PROVIDER

@dataclass(frozen=True)
class _ModelProviderDiagnosticSpec:
    support_level: ProviderDiagnosticSupportLevel
    package_name: str | None
    api_key_envs: tuple[str, ...] = ()
    runtime_config: str = ""

    @property
    def api_key_env_payload(self) -> str | list[str]:
        if len(self.api_key_envs) == 1:
            return self.api_key_envs[0]
        return list(self.api_key_envs)


_EMBEDDING_PROVIDER_SPECS = {
    DEFAULT_EMBEDDING_PROVIDER: _ModelProviderDiagnosticSpec(
        support_level=SUPPORT_DEFAULT,
        package_name=OPENAI_PACKAGE,
        api_key_envs=OPENAI_API_KEY_ENVS,
        runtime_config="RAGCoreConfig.embedding",
    ),
    DEMO_EMBEDDING_PROVIDER: _ModelProviderDiagnosticSpec(
        support_level=SUPPORT_FIRST_PARTY_UTILITY,
        package_name=None,
        runtime_config="RAGCoreConfig.embedding",
    ),
    VOYAGE_PROVIDER: _ModelProviderDiagnosticSpec(
        support_level=SUPPORT_FIRST_PARTY_OPTIONAL,
        package_name=VOYAGE_PACKAGE,
        api_key_envs=VOYAGE_API_KEY_ENVS,
        runtime_config="RAGCoreConfig.embedding",
    ),
    ZEROENTROPY_PROVIDER: _ModelProviderDiagnosticSpec(
        support_level=SUPPORT_FIRST_PARTY_OPTIONAL,
        package_name=ZEROENTROPY_PACKAGE,
        api_key_envs=ZEROENTROPY_API_KEY_ENVS,
        runtime_config="RAGCoreConfig.embedding",
    ),
}
_RERANKER_PROVIDER_SPECS = {
    DEFAULT_RERANKER_PROVIDER: _ModelProviderDiagnosticSpec(
        support_level=SUPPORT_DEFAULT_NOOP,
        package_name=None,
        runtime_config="RAGCoreConfig.reranker",
    ),
    COHERE_RERANKER_PROVIDER: _ModelProviderDiagnosticSpec(
        support_level=SUPPORT_FIRST_PARTY_OPTIONAL,
        package_name=COHERE_PACKAGE,
        api_key_envs=reranker_api_key_env_names(COHERE_RERANKER_PROVIDER),
        runtime_config="RAGCoreConfig.reranker",
    ),
    VOYAGE_PROVIDER: _ModelProviderDiagnosticSpec(
        support_level=SUPPORT_FIRST_PARTY_OPTIONAL,
        package_name=VOYAGE_PACKAGE,
        api_key_envs=reranker_api_key_env_names(VOYAGE_PROVIDER),
        runtime_config="RAGCoreConfig.reranker",
    ),
    ZEROENTROPY_PROVIDER: _ModelProviderDiagnosticSpec(
        support_level=SUPPORT_FIRST_PARTY_OPTIONAL,
        package_name=ZEROENTROPY_PACKAGE,
        api_key_envs=reranker_api_key_env_names(ZEROENTROPY_PROVIDER),
        runtime_config="RAGCoreConfig.reranker",
    ),
}
EMBEDDING_PROVIDER_ORDER = tuple(_EMBEDDING_PROVIDER_SPECS)
RERANKER_PROVIDER_ORDER = tuple(_RERANKER_PROVIDER_SPECS)


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
        EMBEDDING_PROVIDER_CATEGORY: describe_embedding_provider_diagnostics(
            config=config,
            embedding_dimensions=embedding_dimensions,
        ),
        SPARSE_PROVIDER_CATEGORY: describe_sparse_provider_diagnostics(
            runtime_provider=sparse_provider_name,
        ),
        RERANKER_PROVIDER_CATEGORY: describe_reranker_provider_diagnostics(
            config=config
        ),
        OCR_PROVIDER_CATEGORY: describe_ocr_provider_diagnostics(
            runtime_provider=ocr_provider_name
        ),
        CONTEXTUALIZER_PROVIDER_CATEGORY: describe_contextualizer_diagnostics(
            runtime_provider=contextualizer_name,
        ),
        EMBEDDING_CACHE_PROVIDER_CATEGORY: describe_embedding_cache_provider_diagnostics(
            config=config,
            runtime_provider=embedding_cache_name,
        ),
        CHUNK_CONTEXT_CACHE_PROVIDER_CATEGORY: describe_chunk_context_cache_provider_diagnostics(
            runtime_provider=chunk_context_cache_name,
        ),
        SEARCH_SIDECAR_PROVIDER_CATEGORY: describe_search_sidecar_provider_diagnostics(
            config=config,
            runtime_provider=search_sidecar_name,
        ),
        EVENT_SINK_PROVIDER_CATEGORY: describe_event_sink_provider_diagnostics(
            runtime_provider=event_sink_name,
        ),
    }


def describe_embedding_provider_diagnostics(
    *,
    config: RAGCoreConfig,
    embedding_dimensions: int,
) -> dict[str, object]:
    configured = _normalize(config.embedding.provider) or DEFAULT_EMBEDDING_PROVIDER
    return {
        FIELD_CONFIGURED: configured,
        FIELD_REGISTERED: list(EMBEDDING_PROVIDERS.names()),
        FIELD_PROVIDERS: {
            provider: _embedding_provider_diagnostics(
                provider,
                spec=spec,
                config=config,
                configured=configured,
                embedding_dimensions=embedding_dimensions,
            )
            for provider, spec in _EMBEDDING_PROVIDER_SPECS.items()
        },
    }


def describe_reranker_provider_diagnostics(
    *,
    config: RAGCoreConfig,
) -> dict[str, object]:
    configured = _normalize(config.reranker.provider) or DEFAULT_RERANKER_PROVIDER
    effective, fallback_reason = resolve_reranker_provider(
        configured,
        api_key=config.reranker.api_key,
    )
    return {
        FIELD_CONFIGURED: configured,
        "effective": effective,
        "fallback_reason": fallback_reason,
        FIELD_REGISTERED: list(RERANKER_PROVIDERS.names()),
        FIELD_PROVIDERS: {
            provider: _reranker_provider_diagnostics(
                provider,
                spec=spec,
                config=config,
                configured=configured,
            )
            for provider, spec in _RERANKER_PROVIDER_SPECS.items()
        },
    }


def _embedding_provider_diagnostics(
    provider: str,
    *,
    spec: _ModelProviderDiagnosticSpec,
    config: RAGCoreConfig,
    configured: str,
    embedding_dimensions: int,
) -> dict[str, object]:
    selected = provider == configured
    model_spec = (
        get_embedding_model_spec(provider, config.embedding.model) if selected else None
    )
    payload: dict[str, object] = {
        FIELD_SUPPORT_LEVEL: spec.support_level,
        FIELD_CONFIGURED: selected,
        FIELD_PACKAGE_AVAILABLE: _package_available(spec),
        FIELD_RUNTIME_CONFIG: spec.runtime_config,
    }
    if spec.api_key_envs:
        payload.update(
            {
                FIELD_API_KEY_ENV: spec.api_key_env_payload,
                FIELD_API_KEY_CONFIGURED: _api_key_configured(
                    spec,
                    selected=selected,
                    explicit_key=config.embedding.api_key,
                ),
            }
        )
    if selected:
        payload.update(
            {
                "model": config.embedding.model,
                "dimensions": embedding_dimensions,
                "model_known": model_spec is not None,
                "dimensions_override": config.embedding.dimensions is not None,
                "batch_size": config.embedding.batch_size,
                "base_url_configured": bool(
                    normalize_optional_provider_config_value(config.embedding.base_url)
                ),
            }
        )
    if model_spec is not None:
        payload.update(
            {
                "default_dimensions": model_spec.default_dimensions,
                "max_dimensions": model_spec.max_dimensions,
                "allowed_dimensions": list(model_spec.allowed_dimensions or ()),
                "supports_dimensions_override": model_spec.supports_dimensions_override,
            }
        )
    return payload


def _reranker_provider_diagnostics(
    provider: str,
    *,
    spec: _ModelProviderDiagnosticSpec,
    config: RAGCoreConfig,
    configured: str,
) -> dict[str, object]:
    selected = provider == configured
    payload: dict[str, object] = {
        FIELD_SUPPORT_LEVEL: spec.support_level,
        FIELD_CONFIGURED: selected,
        FIELD_PACKAGE_AVAILABLE: _package_available(spec),
        FIELD_RUNTIME_CONFIG: spec.runtime_config,
    }
    if spec.api_key_envs:
        payload.update(
            {
                FIELD_API_KEY_ENV: spec.api_key_env_payload,
                FIELD_API_KEY_CONFIGURED: _api_key_configured(
                    spec,
                    selected=selected,
                    explicit_key=config.reranker.api_key,
                ),
            }
        )
    if selected:
        payload["model"] = config.reranker.model
    return payload


def _api_key_configured(
    spec: _ModelProviderDiagnosticSpec,
    *,
    selected: bool,
    explicit_key: str | None,
) -> bool:
    selected_explicit_key = explicit_key if selected else None
    return api_key_configured(
        spec.api_key_envs,
        explicit_key=selected_explicit_key,
    )


def _package_available(spec: _ModelProviderDiagnosticSpec) -> bool:
    if spec.package_name is None:
        return True
    return importlib.util.find_spec(spec.package_name) is not None


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
