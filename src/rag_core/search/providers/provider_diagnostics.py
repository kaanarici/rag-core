"""Provider readiness diagnostics used by doctor surfaces."""

from __future__ import annotations

from rag_core.config import DEFAULT_EMBEDDING_PROVIDER, DEFAULT_RERANKER_PROVIDER
from rag_core.config.env_access import get_env_stripped
from rag_core.core_models import Config
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
from rag_core.provider_api_keys import ANTHROPIC_API_KEY_ENVS, GEMINI_API_KEY_ENVS, MISTRAL_API_KEY_ENVS
from rag_core.provider_package_names import ANTHROPIC_PACKAGE, FASTEMBED_PACKAGE
from rag_core.search.lexical_sidecar import (
    PORTABLE_LEXICAL_SIDECAR_PROVIDER,
    SEARCH_SIDECAR_PROVIDER_ORDER,
)
from rag_core.search.sparse_channels import (
    PRIMARY_SPARSE_CHANNEL,
    SECONDARY_SPARSE_CHANNEL,
)

from .cache_sqlite import CACHE_PROVIDER_ORDER, NO_CACHE_PROVIDER
from .diagnostic_support import (
    FIELD_API_KEY_CONFIGURED,
    FIELD_API_KEY_ENV,
    FIELD_CONFIGURED,
    FIELD_PACKAGE_AVAILABLE,
    FIELD_PROVIDERS,
    FIELD_READINESS_SCOPE,
    FIELD_REGISTERED,
    FIELD_RUNTIME_CONFIG,
    FIELD_MATURITY,
    READINESS_INSTALLED_AND_CONFIGURED,
    MATURITY_DEFAULT,
    MATURITY_DISABLED,
    MATURITY_OPTIONAL,
    MATURITY_UTILITY,
)
from .event_sink_category_diagnostics import describe_event_sink_provider_diagnostics
from .model_provider_specs import (
    EMBEDDING_PROVIDER_SPECS,
    RERANKER_PROVIDER_SPECS,
    embedding_provider_diagnostics,
    reranker_provider_diagnostics,
)
from .provider_category_helpers import (
    api_env_configured,
    category_diagnostics,
    normalize,
    normalize_runtime_provider,
    package_available,
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
from .registry import (
    CHUNK_CONTEXT_CACHES,
    CONTEXTUALIZER_PROVIDERS,
    EMBEDDING_CACHES,
    EMBEDDING_PROVIDERS,
    OCR_PROVIDERS,
    RERANKER_PROVIDERS,
    SEARCH_SIDECARS,
    SPARSE_EMBEDDERS,
)
from .reranker_resolution import resolve_reranker_provider
from .sparse import (
    DEFAULT_SPARSE_EMBEDDER_PROVIDER,
    SPARSE_EMBEDDER_PROVIDER_ORDER,
    SPARSE_LOAD_NOT_CHECKED_BY_DOCTOR,
    SPLADE_LOAD_UNKNOWN_UNTIL_RUN,
)

EMBEDDING_PROVIDER_ORDER = tuple(EMBEDDING_PROVIDER_SPECS)
RERANKER_PROVIDER_ORDER = tuple(RERANKER_PROVIDER_SPECS)

_SPARSE_MODEL_ENVS = (
    "SPARSE_EMBEDDING_MODEL",
    "SPARSE_EMBEDDING_MODEL_BM25",
    "SPARSE_EMBEDDING_MODEL_SPLADE",
)
_PACKAGE_BY_PROVIDER = {
    DEFAULT_SPARSE_EMBEDDER_PROVIDER: FASTEMBED_PACKAGE,
    ANTHROPIC_CONTEXTUALIZER_ID: ANTHROPIC_PACKAGE,
}


def describe_model_provider_diagnostics(
    *,
    config: Config,
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
            config=config,
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
    config: Config,
    embedding_dimensions: int,
) -> dict[str, object]:
    configured = normalize(config.embedding.provider) or DEFAULT_EMBEDDING_PROVIDER
    return {
        FIELD_CONFIGURED: configured,
        FIELD_REGISTERED: list(EMBEDDING_PROVIDERS.names()),
        FIELD_PROVIDERS: {
            provider: embedding_provider_diagnostics(
                provider,
                spec=spec,
                config=config,
                configured=configured,
                embedding_dimensions=embedding_dimensions,
            )
            for provider, spec in EMBEDDING_PROVIDER_SPECS.items()
        },
    }


def describe_reranker_provider_diagnostics(
    *,
    config: Config,
) -> dict[str, object]:
    configured = normalize(config.reranker.provider) or DEFAULT_RERANKER_PROVIDER
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
            provider: reranker_provider_diagnostics(
                provider,
                spec=spec,
                config=config,
                configured=configured,
            )
            for provider, spec in RERANKER_PROVIDER_SPECS.items()
        },
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
            FIELD_MATURITY: MATURITY_DEFAULT,
            FIELD_CONFIGURED: configured == DEFAULT_SPARSE_EMBEDDER_PROVIDER,
            FIELD_PACKAGE_AVAILABLE: package_available(
                DEFAULT_SPARSE_EMBEDDER_PROVIDER,
                packages_by_provider=_PACKAGE_BY_PROVIDER,
            ),
            FIELD_RUNTIME_CONFIG: "Engine(..., sparse_embedder=...) or default",
            FIELD_READINESS_SCOPE: READINESS_INSTALLED_AND_CONFIGURED,
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
    return category_diagnostics(
        configured=configured,
        registered=list(SPARSE_EMBEDDERS.names()),
        providers=providers,
        known=SPARSE_EMBEDDER_PROVIDER_ORDER,
    )


def describe_ocr_provider_diagnostics(
    *,
    runtime_provider: str | None = None,
) -> dict[str, object]:
    configured = normalize_runtime_provider(runtime_provider)
    providers: dict[str, object] = {
        MISTRAL_OCR_PROVIDER: {
            FIELD_MATURITY: MATURITY_OPTIONAL,
            FIELD_CONFIGURED: configured == MISTRAL_OCR_PROVIDER,
            FIELD_PACKAGE_AVAILABLE: True,
            FIELD_API_KEY_ENV: MISTRAL_API_KEY_ENVS[0],
            FIELD_API_KEY_CONFIGURED: api_env_configured(
                MISTRAL_API_KEY_ENVS
            ),
            "supports_page_selection": True,
            FIELD_RUNTIME_CONFIG: "Engine(..., ocr_provider=...)",
        },
        GEMINI_OCR_PROVIDER: {
            FIELD_MATURITY: MATURITY_OPTIONAL,
            FIELD_CONFIGURED: configured == GEMINI_OCR_PROVIDER,
            FIELD_PACKAGE_AVAILABLE: True,
            FIELD_API_KEY_ENV: list(GEMINI_API_KEY_ENVS),
            FIELD_API_KEY_CONFIGURED: api_env_configured(GEMINI_API_KEY_ENVS),
            "supports_page_selection": False,
            FIELD_RUNTIME_CONFIG: "Engine(..., ocr_provider=...)",
        },
    }
    return category_diagnostics(
        configured=configured,
        registered=list(OCR_PROVIDERS.names()),
        providers=providers,
        known=OCR_PROVIDER_ORDER,
    )


def describe_contextualizer_diagnostics(
    *,
    config: Config | None = None,
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
            FIELD_MATURITY: MATURITY_DISABLED,
            FIELD_CONFIGURED: configured == NOOP_CONTEXTUALIZER_ID,
            FIELD_PACKAGE_AVAILABLE: True,
            FIELD_RUNTIME_CONFIG: (
                "Config.contextualizer.enabled=False or "
                "Engine(..., chunk_contextualizer=None)"
            ),
        },
        ANTHROPIC_CONTEXTUALIZER_ID: {
            FIELD_MATURITY: MATURITY_OPTIONAL,
            FIELD_CONFIGURED: configured == ANTHROPIC_CONTEXTUALIZER_ID,
            FIELD_PACKAGE_AVAILABLE: package_available(
                ANTHROPIC_CONTEXTUALIZER_ID,
                packages_by_provider=_PACKAGE_BY_PROVIDER,
            ),
            FIELD_API_KEY_ENV: ANTHROPIC_API_KEY_ENVS[0],
            FIELD_API_KEY_CONFIGURED: api_env_configured(ANTHROPIC_API_KEY_ENVS),
            FIELD_RUNTIME_CONFIG: (
                "Config.contextualizer or "
                "Engine(..., chunk_contextualizer=...)"
            ),
        },
    }
    return category_diagnostics(
        configured=configured,
        registered=list(CONTEXTUALIZER_PROVIDERS.names()),
        providers=providers,
        known=CONTEXTUALIZER_PROVIDER_ORDER,
    )


def describe_embedding_cache_provider_diagnostics(
    *,
    config: Config,
    runtime_provider: str | None = None,
) -> dict[str, object]:
    configured = normalize_runtime_provider(
        runtime_provider,
        default=normalize(config.ingest.embedding_cache_provider)
        or NO_CACHE_PROVIDER,
    ) or NO_CACHE_PROVIDER
    return _describe_cache_provider_diagnostics(
        configured,
        registered=list(EMBEDDING_CACHES.names()),
    )


def describe_chunk_context_cache_provider_diagnostics(
    *,
    runtime_provider: str | None = None,
) -> dict[str, object]:
    configured = normalize_runtime_provider(
        runtime_provider,
        default=NO_CACHE_PROVIDER,
    ) or NO_CACHE_PROVIDER
    return _describe_cache_provider_diagnostics(
        configured,
        registered=list(CHUNK_CONTEXT_CACHES.names()),
    )


def describe_search_sidecar_provider_diagnostics(
    *,
    config: Config,
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
            FIELD_MATURITY: MATURITY_UTILITY,
            FIELD_CONFIGURED: configured == PORTABLE_LEXICAL_SIDECAR_PROVIDER,
            FIELD_PACKAGE_AVAILABLE: True,
            FIELD_RUNTIME_CONFIG: (
                "Config.ingest.lexical_search_provider or "
                "Engine(..., search_sidecar=...)"
            ),
        }
    }
    return category_diagnostics(
        configured=configured or None,
        registered=list(SEARCH_SIDECARS.names()),
        providers=providers,
        known=SEARCH_SIDECAR_PROVIDER_ORDER,
    )


def _describe_cache_provider_diagnostics(
    configured: str,
    *,
    registered: list[str],
) -> dict[str, object]:
    providers: dict[str, object] = {
        provider: _cache_provider_diagnostics(provider, configured=configured)
        for provider in CACHE_PROVIDER_ORDER
    }
    return category_diagnostics(
        configured=configured,
        registered=registered,
        providers=providers,
        known=CACHE_PROVIDER_ORDER,
    )


def _cache_provider_diagnostics(
    provider: str,
    *,
    configured: str,
) -> dict[str, object]:
    return {
        FIELD_MATURITY: (
            MATURITY_DISABLED
            if provider == NO_CACHE_PROVIDER
            else MATURITY_UTILITY
        ),
        FIELD_CONFIGURED: provider == configured,
        FIELD_PACKAGE_AVAILABLE: True,
        FIELD_RUNTIME_CONFIG: "registered provider name or direct constructor injection",
    }


def _normalize_contextualizer_provider(
    value: str | None,
    *,
    config: Config | None,
) -> str:
    default = NOOP_CONTEXTUALIZER_ID
    if config is not None and config.contextualizer.enabled:
        default = config.contextualizer.provider
    configured = normalize_runtime_provider(value, default=default)
    if configured == CONTEXTUALIZER_DISABLED_ALIAS:
        return NOOP_CONTEXTUALIZER_ID
    return configured or NOOP_CONTEXTUALIZER_ID


__all__ = [
    "EMBEDDING_PROVIDER_ORDER",
    "RERANKER_PROVIDER_ORDER",
    "describe_chunk_context_cache_provider_diagnostics",
    "describe_contextualizer_diagnostics",
    "describe_embedding_cache_provider_diagnostics",
    "describe_embedding_provider_diagnostics",
    "describe_event_sink_provider_diagnostics",
    "describe_model_provider_diagnostics",
    "describe_ocr_provider_diagnostics",
    "describe_reranker_provider_diagnostics",
    "describe_search_sidecar_provider_diagnostics",
    "describe_sparse_provider_diagnostics",
]
