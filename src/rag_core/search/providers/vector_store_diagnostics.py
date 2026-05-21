"""Vector-store diagnostics used by CLI/runtime doctor surfaces."""

from __future__ import annotations

from rag_core.cli_inputs import cli_redacted_url, cli_store_location_label
from rag_core.core_models import RAGCoreConfig
from rag_core.core_runtime import describe_query_plan_capabilities

from .query_plan_capabilities import QDRANT_QUERY_PLAN_CAPABILITIES
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
