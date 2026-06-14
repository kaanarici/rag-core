"""Diagnostics for non-model provider categories."""

from __future__ import annotations

from rag_core.config.env_access import get_env_stripped
from rag_core.core_models import RAGCoreConfig
from rag_core.documents.contextualizer_provider_names import (
    ANTHROPIC_CONTEXTUALIZER_ID,
    CONTEXTUALIZER_DISABLED_ALIAS,
    CONTEXTUALIZER_PROVIDER_ORDER,
    NOOP_CONTEXTUALIZER_ID,
)
from rag_core.documents.ocr_provider_names import (
    GEMINI_OCR_PROVIDER,
    MISTRAL_OCR_PROVIDER,
    OCR_PROVIDER_ORDER,
)
from rag_core.provider_api_keys import (
    ANTHROPIC_API_KEY_ENVS,
    GEMINI_API_KEY_ENVS,
    MISTRAL_API_KEY_ENVS,
)
from rag_core.provider_package_names import ANTHROPIC_PACKAGE, FASTEMBED_PACKAGE
from rag_core.search.lexical_sidecar import (
    PORTABLE_LEXICAL_SIDECAR_PROVIDER,
    SEARCH_SIDECAR_PROVIDER_ORDER,
)
from rag_core.search.sparse_channels import (
    PRIMARY_SPARSE_CHANNEL,
    SECONDARY_SPARSE_CHANNEL,
)

from .diagnostic_support import (
    FIELD_API_KEY_CONFIGURED,
    FIELD_API_KEY_ENV,
    FIELD_CONFIGURED,
    FIELD_PACKAGE_AVAILABLE,
    FIELD_PROVIDERS,
    FIELD_READINESS_SCOPE,
    FIELD_REGISTERED,
    FIELD_RUNTIME_CONFIG,
    FIELD_SUPPORT_LEVEL,
    READINESS_PACKAGE_AND_ENV,
    SUPPORT_DEFAULT,
    SUPPORT_DEFAULT_NOOP,
    SUPPORT_FIRST_PARTY_OPTIONAL,
    SUPPORT_FIRST_PARTY_UTILITY,
)
from .cache_provider_names import CACHE_PROVIDER_ORDER, NO_CACHE_PROVIDER
from .event_sink_category_diagnostics import describe_event_sink_provider_diagnostics
from .provider_category_helpers import (
    add_injected_provider,
    api_env_configured,
    normalize,
    normalize_runtime_provider,
    package_available,
)
from .registry import (
    CHUNK_CONTEXT_CACHES,
    CONTEXTUALIZER_PROVIDERS,
    EMBEDDING_CACHES,
    OCR_PROVIDERS,
    SEARCH_SIDECARS,
    SPARSE_EMBEDDERS,
)
from .sparse import (
    DEFAULT_SPARSE_EMBEDDER_PROVIDER,
    SPARSE_EMBEDDER_PROVIDER_ORDER,
    SPARSE_LOAD_NOT_CHECKED_BY_DOCTOR,
    SPLADE_LOAD_UNKNOWN_UNTIL_RUN,
)

_SPARSE_MODEL_ENVS = (
    "SPARSE_EMBEDDING_MODEL",
    "SPARSE_EMBEDDING_MODEL_BM25",
    "SPARSE_EMBEDDING_MODEL_SPLADE",
)
_PACKAGE_BY_PROVIDER = {
    DEFAULT_SPARSE_EMBEDDER_PROVIDER: FASTEMBED_PACKAGE,
    ANTHROPIC_CONTEXTUALIZER_ID: ANTHROPIC_PACKAGE,
}


def describe_sparse_provider_diagnostics(
    *,
    runtime_provider: str | None = None,
) -> dict[str, object]:
    configured = normalize_runtime_provider(
        runtime_provider,
        default=DEFAULT_SPARSE_EMBEDDER_PROVIDER,
    )
    providers: dict[str, object] = {
        DEFAULT_SPARSE_EMBEDDER_PROVIDER: {
            FIELD_SUPPORT_LEVEL: SUPPORT_DEFAULT,
            FIELD_CONFIGURED: configured == DEFAULT_SPARSE_EMBEDDER_PROVIDER,
            FIELD_PACKAGE_AVAILABLE: package_available(
                DEFAULT_SPARSE_EMBEDDER_PROVIDER,
                packages_by_provider=_PACKAGE_BY_PROVIDER,
            ),
            FIELD_RUNTIME_CONFIG: "RAGCore(..., sparse_embedder=...) or default",
            FIELD_READINESS_SCOPE: READINESS_PACKAGE_AND_ENV,
            "channels": {
                PRIMARY_SPARSE_CHANNEL: {
                    "enabled": True,
                    "load_status": SPARSE_LOAD_NOT_CHECKED_BY_DOCTOR,
                },
                SECONDARY_SPARSE_CHANNEL: {
                    "enabled_by_default": True,
                    "live_ready": None,
                    "load_status": SPLADE_LOAD_UNKNOWN_UNTIL_RUN,
                },
            },
            "model_env_configured": {
                env_name: bool(get_env_stripped(env_name))
                for env_name in _SPARSE_MODEL_ENVS
            },
        }
    }
    add_injected_provider(
        providers,
        configured,
        known=SPARSE_EMBEDDER_PROVIDER_ORDER,
    )
    return {
        FIELD_CONFIGURED: configured,
        FIELD_REGISTERED: list(SPARSE_EMBEDDERS.names()),
        FIELD_PROVIDERS: providers,
    }


def describe_ocr_provider_diagnostics(
    *,
    runtime_provider: str | None = None,
) -> dict[str, object]:
    configured = normalize_runtime_provider(runtime_provider)
    providers: dict[str, object] = {
        MISTRAL_OCR_PROVIDER: {
            FIELD_SUPPORT_LEVEL: SUPPORT_FIRST_PARTY_OPTIONAL,
            FIELD_CONFIGURED: configured == MISTRAL_OCR_PROVIDER,
            FIELD_PACKAGE_AVAILABLE: True,
            FIELD_API_KEY_ENV: MISTRAL_API_KEY_ENVS[0],
            FIELD_API_KEY_CONFIGURED: api_env_configured(
                MISTRAL_API_KEY_ENVS
            ),
            "supports_page_selection": True,
            FIELD_RUNTIME_CONFIG: "RAGCore(..., ocr_provider=...)",
        },
        GEMINI_OCR_PROVIDER: {
            FIELD_SUPPORT_LEVEL: SUPPORT_FIRST_PARTY_OPTIONAL,
            FIELD_CONFIGURED: configured == GEMINI_OCR_PROVIDER,
            FIELD_PACKAGE_AVAILABLE: True,
            FIELD_API_KEY_ENV: list(GEMINI_API_KEY_ENVS),
            FIELD_API_KEY_CONFIGURED: api_env_configured(GEMINI_API_KEY_ENVS),
            "supports_page_selection": False,
            FIELD_RUNTIME_CONFIG: "RAGCore(..., ocr_provider=...)",
        },
    }
    add_injected_provider(providers, configured, known=OCR_PROVIDER_ORDER)
    return {
        FIELD_CONFIGURED: configured,
        FIELD_REGISTERED: list(OCR_PROVIDERS.names()),
        FIELD_PROVIDERS: providers,
    }


def describe_contextualizer_diagnostics(
    *,
    config: RAGCoreConfig | None = None,
    runtime_provider: str | None = None,
) -> dict[str, object]:
    configured = _normalize_contextualizer_provider(
        runtime_provider,
        config=config,
    )
    if configured.startswith(f"{ANTHROPIC_CONTEXTUALIZER_ID}:"):
        configured = ANTHROPIC_CONTEXTUALIZER_ID
    providers: dict[str, object] = {
        NOOP_CONTEXTUALIZER_ID: {
            FIELD_SUPPORT_LEVEL: SUPPORT_DEFAULT_NOOP,
            FIELD_CONFIGURED: configured == NOOP_CONTEXTUALIZER_ID,
            FIELD_PACKAGE_AVAILABLE: True,
            FIELD_RUNTIME_CONFIG: (
                "RAGCoreConfig.contextualizer.enabled=False or "
                "RAGCore(..., chunk_contextualizer=None)"
            ),
        },
        ANTHROPIC_CONTEXTUALIZER_ID: {
            FIELD_SUPPORT_LEVEL: SUPPORT_FIRST_PARTY_OPTIONAL,
            FIELD_CONFIGURED: configured == ANTHROPIC_CONTEXTUALIZER_ID,
            FIELD_PACKAGE_AVAILABLE: package_available(
                ANTHROPIC_CONTEXTUALIZER_ID,
                packages_by_provider=_PACKAGE_BY_PROVIDER,
            ),
            FIELD_API_KEY_ENV: ANTHROPIC_API_KEY_ENVS[0],
            FIELD_API_KEY_CONFIGURED: api_env_configured(ANTHROPIC_API_KEY_ENVS),
            FIELD_RUNTIME_CONFIG: (
                "RAGCoreConfig.contextualizer or "
                "RAGCore(..., chunk_contextualizer=...)"
            ),
        },
    }
    add_injected_provider(
        providers,
        configured,
        known=CONTEXTUALIZER_PROVIDER_ORDER,
    )
    return {
        FIELD_CONFIGURED: configured,
        FIELD_REGISTERED: list(CONTEXTUALIZER_PROVIDERS.names()),
        FIELD_PROVIDERS: providers,
    }


def describe_embedding_cache_provider_diagnostics(
    *,
    config: RAGCoreConfig,
    runtime_provider: str | None = None,
) -> dict[str, object]:
    configured = normalize_runtime_provider(
        runtime_provider,
        default=normalize(config.ingest.embedding_cache_provider)
        or NO_CACHE_PROVIDER,
    ) or NO_CACHE_PROVIDER
    providers: dict[str, object] = {
        provider: _cache_provider_diagnostics(provider, configured=configured)
        for provider in CACHE_PROVIDER_ORDER
    }
    add_injected_provider(providers, configured, known=CACHE_PROVIDER_ORDER)
    return {
        FIELD_CONFIGURED: configured,
        FIELD_REGISTERED: list(EMBEDDING_CACHES.names()),
        FIELD_PROVIDERS: providers,
    }


def describe_chunk_context_cache_provider_diagnostics(
    *,
    runtime_provider: str | None = None,
) -> dict[str, object]:
    configured = normalize_runtime_provider(
        runtime_provider,
        default=NO_CACHE_PROVIDER,
    ) or NO_CACHE_PROVIDER
    providers: dict[str, object] = {
        provider: _cache_provider_diagnostics(provider, configured=configured)
        for provider in CACHE_PROVIDER_ORDER
    }
    add_injected_provider(providers, configured, known=CACHE_PROVIDER_ORDER)
    return {
        FIELD_CONFIGURED: configured,
        FIELD_REGISTERED: list(CHUNK_CONTEXT_CACHES.names()),
        FIELD_PROVIDERS: providers,
    }


def describe_search_sidecar_provider_diagnostics(
    *,
    config: RAGCoreConfig,
    runtime_provider: str | None = None,
) -> dict[str, object]:
    configured = normalize_runtime_provider(runtime_provider)
    if not configured:
        configured = normalize(config.ingest.lexical_search_provider)
    if not configured:
        configured = None
    if not configured and config.ingest.enable_lexical_search:
        configured = PORTABLE_LEXICAL_SIDECAR_PROVIDER
    providers: dict[str, object] = {
        PORTABLE_LEXICAL_SIDECAR_PROVIDER: {
            FIELD_SUPPORT_LEVEL: SUPPORT_FIRST_PARTY_UTILITY,
            FIELD_CONFIGURED: configured == PORTABLE_LEXICAL_SIDECAR_PROVIDER,
            FIELD_PACKAGE_AVAILABLE: True,
            FIELD_RUNTIME_CONFIG: (
                "RAGCoreConfig.ingest.lexical_search_provider or "
                "RAGCore(..., search_sidecar=...)"
            ),
        }
    }
    add_injected_provider(
        providers,
        configured,
        known=SEARCH_SIDECAR_PROVIDER_ORDER,
    )
    return {
        FIELD_CONFIGURED: configured or None,
        FIELD_REGISTERED: list(SEARCH_SIDECARS.names()),
        FIELD_PROVIDERS: providers,
    }


def _cache_provider_diagnostics(
    provider: str,
    *,
    configured: str,
) -> dict[str, object]:
    return {
        FIELD_SUPPORT_LEVEL: (
            SUPPORT_DEFAULT_NOOP
            if provider == NO_CACHE_PROVIDER
            else SUPPORT_FIRST_PARTY_UTILITY
        ),
        FIELD_CONFIGURED: provider == configured,
        FIELD_PACKAGE_AVAILABLE: True,
        FIELD_RUNTIME_CONFIG: "registered provider name or direct constructor injection",
    }


def _normalize_contextualizer_provider(
    value: str | None,
    *,
    config: RAGCoreConfig | None,
) -> str:
    default = NOOP_CONTEXTUALIZER_ID
    if config is not None and config.contextualizer.enabled:
        default = config.contextualizer.provider
    configured = normalize_runtime_provider(value, default=default)
    if configured == CONTEXTUALIZER_DISABLED_ALIAS:
        return NOOP_CONTEXTUALIZER_ID
    return configured or NOOP_CONTEXTUALIZER_ID


__all__ = [
    "describe_embedding_cache_provider_diagnostics",
    "describe_chunk_context_cache_provider_diagnostics",
    "describe_contextualizer_diagnostics",
    "describe_event_sink_provider_diagnostics",
    "describe_ocr_provider_diagnostics",
    "describe_search_sidecar_provider_diagnostics",
    "describe_sparse_provider_diagnostics",
]
