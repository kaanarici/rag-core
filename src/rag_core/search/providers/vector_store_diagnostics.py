"""Vector-store diagnostics used by CLI/runtime doctor surfaces."""

from __future__ import annotations

import importlib.util

from rag_core.cli_inputs import cli_redacted_url, cli_store_location_label
from rag_core.config.env_access import get_env_stripped
from rag_core.core_models import RAGCoreConfig
from rag_core.core_runtime import describe_query_plan_capabilities

from .query_plan_capabilities import (
    QDRANT_QUERY_PLAN_CAPABILITIES,
    TURBOPUFFER_QUERY_PLAN_CAPABILITIES,
)
from .registry import VECTOR_STORES


def describe_vector_store_diagnostics(
    *,
    config: RAGCoreConfig,
    collection_name: str,
) -> dict[str, object]:
    return {
        "configured": config.vector_store.provider,
        "default": "qdrant",
        "registered": list(VECTOR_STORES.names()),
        "providers": {
            "qdrant": _qdrant_diagnostics(
                config=config,
                collection_name=collection_name,
            ),
            "turbopuffer": _turbopuffer_diagnostics(config),
        },
    }


def _qdrant_diagnostics(
    *,
    config: RAGCoreConfig,
    collection_name: str,
) -> dict[str, object]:
    return {
        "support_level": "default",
        "configured": config.vector_store.provider == "qdrant",
        "check_store_supported": True,
        "collection_name": (
            collection_name if config.vector_store.provider == "qdrant" else None
        ),
        "url": cli_redacted_url(config.qdrant.url),
        "location": cli_store_location_label(config.qdrant.location),
        "connection_configured": bool(config.qdrant.url or config.qdrant.location),
        "dimension_aware_collection": config.qdrant.dimension_aware_collection,
        "query_plan_scope": "adapter_maximum",
        "query_plan": describe_query_plan_capabilities(QDRANT_QUERY_PLAN_CAPABILITIES),
    }


def _turbopuffer_diagnostics(config: RAGCoreConfig) -> dict[str, object]:
    tp = config.vector_store.turbopuffer
    api_key_configured = bool(tp.api_key or get_env_stripped("TURBOPUFFER_API_KEY"))
    region = tp.region or get_env_stripped("TURBOPUFFER_REGION") or None
    base_url_configured = bool(tp.base_url or get_env_stripped("TURBOPUFFER_BASE_URL"))
    return {
        "support_level": "first_party_optional",
        "configured": config.vector_store.provider == "turbopuffer",
        "check_store_supported": True,
        "extra": "turbopuffer",
        "package_available": importlib.util.find_spec("turbopuffer") is not None,
        "api_key_configured": api_key_configured,
        "namespace": tp.namespace,
        "region": region,
        "base_url_configured": base_url_configured,
        "distance_metric": tp.distance_metric,
        "runtime_config": "RAGCoreConfig.vector_store",
        "query_plan_scope": "adapter_maximum",
        "query_plan": describe_query_plan_capabilities(
            TURBOPUFFER_QUERY_PLAN_CAPABILITIES
        ),
    }
