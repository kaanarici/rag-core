"""Diagnostics for non-model provider categories."""

from __future__ import annotations

import importlib.util

from rag_core.config.env_access import get_env_stripped
from rag_core.core_models import RAGCoreConfig

from .registry import (
    CHUNK_CONTEXT_CACHES,
    EMBEDDING_CACHES,
    OCR_PROVIDERS,
    SEARCH_SIDECARS,
    SPARSE_EMBEDDERS,
)

_CACHE_PROVIDER_ORDER = ("none", "in_memory", "sqlite")
_SPARSE_MODEL_ENVS = (
    "SPARSE_EMBEDDING_MODEL",
    "SPARSE_EMBEDDING_MODEL_BM25",
    "SPARSE_EMBEDDING_MODEL_SPLADE",
)
_PACKAGE_BY_PROVIDER = {
    "fastembed": "fastembed",
    "anthropic": "anthropic",
    "opentelemetry": "opentelemetry.trace",
}
_API_KEY_ENV_BY_PROVIDER = {
    "mistral": "MISTRAL_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
_GEMINI_API_KEY_ENVS = ("GOOGLE_API_KEY", "GEMINI_API_KEY")
_SPARSE_ALIASES = {"fastembedsparseembedder": "fastembed"}
_CONTEXTUALIZER_ALIASES = {
    "noopcontextualizer": "noop",
    "anthropicchunkcontextualizer": "anthropic",
}
_CACHE_ALIASES = {
    "nocache": "none",
    "nochunkcontextcache": "none",
    "inmemorycache": "in_memory",
    "inmemorychunkcontextcache": "in_memory",
    "sqlitecache": "sqlite",
    "sqlitechunkcontextcache": "sqlite",
}
_SEARCH_SIDECAR_ALIASES = {"portablelexicalsidecar": "portable_lexical"}
_EVENT_SINK_ALIASES = {
    "noop": "none",
    "noopsink": "none",
    "loggingsink": "logging",
    "jsonlsink": "jsonl",
    "eventbuffer": "buffer",
    "opentelemetrysink": "opentelemetry",
}
_EVENT_SINK_RUNTIME_CONFIG = {
    "logging": "RAGCore(..., event_sink=LoggingSink(...))",
    "jsonl": "RAGCore(..., event_sink=JsonlSink(...))",
    "buffer": "RAGCore(..., event_sink=EventBuffer())",
}


def describe_sparse_provider_diagnostics(
    *,
    runtime_provider: str | None = None,
) -> dict[str, object]:
    configured = _normalize_runtime_provider(
        runtime_provider,
        aliases=_SPARSE_ALIASES,
        default="fastembed",
    )
    providers: dict[str, object] = {
        "fastembed": {
            "support_level": "default",
            "configured": configured == "fastembed",
            "package_available": _package_available("fastembed"),
            "runtime_config": "RAGCore(..., sparse_embedder=...) or default",
            "readiness_scope": "package_and_env",
            "channels": {
                "bm25": {
                    "enabled": True,
                    "load_status": "not_checked_by_doctor",
                },
                "splade": {
                    "enabled_by_default": True,
                    "live_ready": None,
                    "load_status": "unknown_until_sparse_embedding_runs",
                },
            },
            "model_env_configured": {
                env_name: bool(get_env_stripped(env_name))
                for env_name in _SPARSE_MODEL_ENVS
            },
        }
    }
    _add_injected_provider(providers, configured, known=("fastembed",))
    return {
        "configured": configured,
        "registered": list(SPARSE_EMBEDDERS.names()),
        "providers": providers,
    }


def describe_ocr_provider_diagnostics(
    *,
    runtime_provider: str | None = None,
) -> dict[str, object]:
    configured = _normalize_runtime_provider(runtime_provider)
    providers: dict[str, object] = {
        "mistral": {
            "support_level": "first_party_optional",
            "configured": configured == "mistral",
            "package_available": True,
            "api_key_env": _API_KEY_ENV_BY_PROVIDER["mistral"],
            "api_key_configured": _api_env_configured(
                (_API_KEY_ENV_BY_PROVIDER["mistral"],)
            ),
            "supports_page_selection": True,
            "runtime_config": "RAGCore(..., ocr_provider=...)",
        },
        "gemini": {
            "support_level": "first_party_optional",
            "configured": configured == "gemini",
            "package_available": True,
            "api_key_env": list(_GEMINI_API_KEY_ENVS),
            "api_key_configured": _api_env_configured(_GEMINI_API_KEY_ENVS),
            "supports_page_selection": False,
            "runtime_config": "RAGCore(..., ocr_provider=...)",
        },
    }
    _add_injected_provider(providers, configured, known=("mistral", "gemini"))
    return {
        "configured": configured,
        "registered": list(OCR_PROVIDERS.names()),
        "providers": providers,
    }


def describe_contextualizer_diagnostics(
    *,
    runtime_provider: str | None = None,
) -> dict[str, object]:
    configured = _normalize_runtime_provider(
        runtime_provider,
        aliases=_CONTEXTUALIZER_ALIASES,
        default="none",
    ) or "none"
    if configured.startswith("anthropic:"):
        configured = "anthropic"
    providers: dict[str, object] = {
        "noop": {
            "support_level": "default_noop",
            "configured": configured == "noop" or configured == "none",
            "package_available": True,
            "runtime_config": "RAGCore(..., chunk_contextualizer=None)",
        },
        "anthropic": {
            "support_level": "first_party_optional",
            "configured": configured == "anthropic",
            "package_available": _package_available("anthropic"),
            "api_key_env": _API_KEY_ENV_BY_PROVIDER["anthropic"],
            "api_key_configured": _api_env_configured(
                (_API_KEY_ENV_BY_PROVIDER["anthropic"],)
            ),
            "runtime_config": "RAGCore(..., chunk_contextualizer=...)",
        },
    }
    _add_injected_provider(providers, configured, known=("none", "noop", "anthropic"))
    return {
        "configured": configured,
        "registered": [],
        "providers": providers,
    }


def describe_embedding_cache_provider_diagnostics(
    *,
    config: RAGCoreConfig,
    runtime_provider: str | None = None,
) -> dict[str, object]:
    configured = _normalize_runtime_provider(
        runtime_provider,
        aliases=_CACHE_ALIASES,
        default=_normalize(config.ingest.embedding_cache_provider) or "none",
    ) or "none"
    providers: dict[str, object] = {
        provider: _cache_provider_diagnostics(provider, configured=configured)
        for provider in _CACHE_PROVIDER_ORDER
    }
    _add_injected_provider(providers, configured, known=_CACHE_PROVIDER_ORDER)
    return {
        "configured": configured,
        "registered": list(EMBEDDING_CACHES.names()),
        "providers": providers,
    }


def describe_chunk_context_cache_provider_diagnostics(
    *,
    runtime_provider: str | None = None,
) -> dict[str, object]:
    configured = _normalize_runtime_provider(
        runtime_provider,
        aliases=_CACHE_ALIASES,
        default="none",
    ) or "none"
    providers: dict[str, object] = {
        provider: _cache_provider_diagnostics(provider, configured=configured)
        for provider in _CACHE_PROVIDER_ORDER
    }
    _add_injected_provider(providers, configured, known=_CACHE_PROVIDER_ORDER)
    return {
        "configured": configured,
        "registered": list(CHUNK_CONTEXT_CACHES.names()),
        "providers": providers,
    }


def describe_search_sidecar_provider_diagnostics(
    *,
    config: RAGCoreConfig,
    runtime_provider: str | None = None,
) -> dict[str, object]:
    configured = _normalize_runtime_provider(
        runtime_provider,
        aliases=_SEARCH_SIDECAR_ALIASES,
    )
    if not configured:
        configured = _normalize(config.ingest.lexical_search_provider)
    if not configured:
        configured = None
    if not configured and config.ingest.enable_lexical_search:
        configured = "portable_lexical"
    providers: dict[str, object] = {
        "portable_lexical": {
            "support_level": "first_party_utility",
            "configured": configured == "portable_lexical",
            "package_available": True,
            "runtime_config": (
                "RAGCoreConfig.ingest.lexical_search_provider or "
                "RAGCore(..., search_sidecar=...)"
            ),
        }
    }
    _add_injected_provider(providers, configured, known=("portable_lexical",))
    return {
        "configured": configured or None,
        "registered": list(SEARCH_SIDECARS.names()),
        "providers": providers,
    }


def describe_event_sink_provider_diagnostics(
    *,
    runtime_provider: str | None = None,
) -> dict[str, object]:
    configured = _normalize_runtime_provider(
        runtime_provider,
        aliases=_EVENT_SINK_ALIASES,
        default="none",
    ) or "none"
    providers: dict[str, object] = {
        "none": {
            "support_level": "default_noop",
            "configured": configured == "none",
            "package_available": True,
            "runtime_config": "RAGCore(..., event_sink=None)",
        },
        "logging": _event_sink_diagnostics("logging", configured=configured),
        "jsonl": _event_sink_diagnostics("jsonl", configured=configured),
        "buffer": _event_sink_diagnostics("buffer", configured=configured),
        "opentelemetry": {
            "support_level": "first_party_optional",
            "configured": configured == "opentelemetry",
            "package_available": _package_available("opentelemetry"),
            "runtime_config": "RAGCore(..., event_sink=OpenTelemetrySink())",
        },
    }
    _add_injected_provider(
        providers,
        configured,
        known=("none", "logging", "jsonl", "buffer", "opentelemetry"),
    )
    return {
        "configured": configured,
        "registered": [],
        "providers": providers,
    }


def _cache_provider_diagnostics(
    provider: str,
    *,
    configured: str,
) -> dict[str, object]:
    return {
        "support_level": "default_noop" if provider == "none" else "first_party_utility",
        "configured": provider == configured,
        "package_available": True,
        "runtime_config": "registered provider name or direct constructor injection",
    }


def _event_sink_diagnostics(provider: str, *, configured: str) -> dict[str, object]:
    return {
        "support_level": "first_party_utility",
        "configured": provider == configured,
        "package_available": True,
        "runtime_config": _EVENT_SINK_RUNTIME_CONFIG[provider],
    }


def _add_injected_provider(
    providers: dict[str, object],
    configured: str | None,
    *,
    known: tuple[str, ...],
) -> None:
    if configured is None or configured in known:
        return
    providers[configured] = {
        "support_level": "injected",
        "configured": True,
        "package_available": None,
        "runtime_config": "direct constructor injection",
    }


def _normalize_runtime_provider(
    value: str | None,
    *,
    aliases: dict[str, str] | None = None,
    default: str | None = None,
) -> str | None:
    normalized = _normalize(value)
    if not normalized:
        return default
    compact = normalized.replace("_", "").replace("-", "")
    return (aliases or {}).get(compact, normalized)


def _api_env_configured(
    env_names: tuple[str, ...],
    *,
    explicit_key: str | None = None,
) -> bool:
    return bool((explicit_key or "").strip()) or any(
        bool(get_env_stripped(env_name)) for env_name in env_names
    )


def _package_available(provider: str) -> bool:
    try:
        return importlib.util.find_spec(_PACKAGE_BY_PROVIDER[provider]) is not None
    except ModuleNotFoundError:
        return False


def _normalize(value: str | None) -> str:
    return (value or "").strip().lower()


__all__ = [
    "describe_embedding_cache_provider_diagnostics",
    "describe_chunk_context_cache_provider_diagnostics",
    "describe_contextualizer_diagnostics",
    "describe_event_sink_provider_diagnostics",
    "describe_ocr_provider_diagnostics",
    "describe_search_sidecar_provider_diagnostics",
    "describe_sparse_provider_diagnostics",
]
