"""Vector-store diagnostics used by CLI/runtime doctor surfaces."""

from __future__ import annotations

import importlib.util
from typing import Final, Literal, TypeAlias

from rag_core.config import DEFAULT_VECTOR_STORE_PROVIDER
from rag_core.config.env_access import get_env_stripped
from rag_core.config.vector_store_config import (
    PGVECTOR_DSN_ENV,
    PGVECTOR_VECTOR_STORE_PROVIDER,
    TURBOPUFFER_BASE_URL_ENV,
    TURBOPUFFER_REGION_ENV,
)
from rag_core.core_models import Config
from rag_core.provider_api_keys import (
    QDRANT_API_KEY_ENVS,
    TURBOPUFFER_API_KEY_ENVS,
    api_key_configured,
)
from rag_core.safe_messages import redacted_url, store_location_label

from .diagnostic_support import (
    FIELD_API_KEY_CONFIGURED,
    FIELD_CONFIGURED,
    FIELD_PACKAGE_AVAILABLE,
    FIELD_PROVIDERS,
    FIELD_REGISTERED,
    FIELD_RUNTIME_CONFIG,
    FIELD_MATURITY,
)
from .vector_store_capabilities import (
    BUILTIN_VECTOR_STORE_PROVIDER_ORDER,
    MEMORY_VECTOR_STORE_PROVIDER_SPEC,
    PGVECTOR_VECTOR_STORE_PROVIDER_SPEC,
    QDRANT_VECTOR_STORE_PROVIDER_SPEC,
    TURBOPUFFER_VECTOR_STORE_PROVIDER_SPEC,
    VectorStoreCapabilitySpec,
    describe_metadata_filter_capabilities,
    describe_query_plan_capabilities,
)
from .registry import VECTOR_STORES

VectorStoreRuntimeValidation: TypeAlias = Literal[
    "not_requested",
    "healthy",
    "failed",
]
VectorStoreQueryPlanScope: TypeAlias = Literal["adapter_maximum"]

VECTOR_STORE_RUNTIME_NOT_REQUESTED: Final[VectorStoreRuntimeValidation] = (
    "not_requested"
)
VECTOR_STORE_RUNTIME_HEALTHY: Final[VectorStoreRuntimeValidation] = "healthy"
VECTOR_STORE_RUNTIME_FAILED: Final[VectorStoreRuntimeValidation] = "failed"
VECTOR_STORE_QUERY_PLAN_SCOPE_ADAPTER_MAXIMUM: Final[VectorStoreQueryPlanScope] = (
    "adapter_maximum"
)


def describe_vector_store_diagnostics(
    *,
    config: Config,
    collection_name: str,
) -> dict[str, object]:
    return {
        FIELD_CONFIGURED: config.vector_store.provider,
        "default": DEFAULT_VECTOR_STORE_PROVIDER,
        FIELD_REGISTERED: list(VECTOR_STORES.names()),
        FIELD_PROVIDERS: {
            QDRANT_VECTOR_STORE_PROVIDER_SPEC.name: _qdrant_diagnostics(
                config=config,
                collection_name=collection_name,
            ),
            PGVECTOR_VECTOR_STORE_PROVIDER_SPEC.name: _pgvector_diagnostics(
                config=config,
                collection_name=collection_name,
            ),
            TURBOPUFFER_VECTOR_STORE_PROVIDER_SPEC.name: _turbopuffer_diagnostics(
                config
            ),
            MEMORY_VECTOR_STORE_PROVIDER_SPEC.name: _memory_diagnostics(),
        },
        "provider_order": BUILTIN_VECTOR_STORE_PROVIDER_ORDER,
    }


def _qdrant_diagnostics(
    *,
    config: Config,
    collection_name: str,
) -> dict[str, object]:
    qdrant_api_key_configured = api_key_configured(
        QDRANT_API_KEY_ENVS,
        explicit_key=config.qdrant.api_key,
        get_env=get_env_stripped,
    )
    return {
        FIELD_MATURITY: QDRANT_VECTOR_STORE_PROVIDER_SPEC.diagnostic_maturity,
        FIELD_CONFIGURED: config.vector_store.provider == QDRANT_VECTOR_STORE_PROVIDER_SPEC.name,
        FIELD_PACKAGE_AVAILABLE: True,
        FIELD_API_KEY_CONFIGURED: qdrant_api_key_configured,
        "credential_required": False,
        "runtime_validated": False,
        "runtime_validation": VECTOR_STORE_RUNTIME_NOT_REQUESTED,
        "check_store_supported": True,
        "collection_name": (
            collection_name
            if config.vector_store.provider == QDRANT_VECTOR_STORE_PROVIDER_SPEC.name
            else None
        ),
        "url": redacted_url(config.qdrant.url),
        "location": store_location_label(config.qdrant.location),
        "connection_configured": bool(config.qdrant.url or config.qdrant.location),
        "dimension_aware_collection": config.qdrant.dimension_aware_collection,
        "query_plan_scope": VECTOR_STORE_QUERY_PLAN_SCOPE_ADAPTER_MAXIMUM,
        **_capability_payload(QDRANT_VECTOR_STORE_PROVIDER_SPEC.capabilities),
    }


def _pgvector_diagnostics(
    *,
    config: Config,
    collection_name: str,
) -> dict[str, object]:
    pgvector = config.vector_store.pgvector
    return {
        FIELD_MATURITY: PGVECTOR_VECTOR_STORE_PROVIDER_SPEC.diagnostic_maturity,
        FIELD_CONFIGURED: config.vector_store.provider == PGVECTOR_VECTOR_STORE_PROVIDER,
        FIELD_PACKAGE_AVAILABLE: (
            importlib.util.find_spec("asyncpg") is not None
            and importlib.util.find_spec("pgvector") is not None
        ),
        FIELD_API_KEY_CONFIGURED: False,
        "credential_required": False,
        "runtime_validated": False,
        "runtime_validation": VECTOR_STORE_RUNTIME_NOT_REQUESTED,
        "check_store_supported": True,
        "extra": PGVECTOR_VECTOR_STORE_PROVIDER_SPEC.name,
        "dsn_configured": bool(pgvector.dsn or get_env_stripped(PGVECTOR_DSN_ENV)),
        "schema": pgvector.schema,
        "table": collection_name,
        FIELD_RUNTIME_CONFIG: "Config.vector_store",
        "query_plan_scope": VECTOR_STORE_QUERY_PLAN_SCOPE_ADAPTER_MAXIMUM,
        **_capability_payload(PGVECTOR_VECTOR_STORE_PROVIDER_SPEC.capabilities),
    }


def _turbopuffer_diagnostics(config: Config) -> dict[str, object]:
    tp = config.vector_store.turbopuffer
    package_available = importlib.util.find_spec("turbopuffer") is not None
    turbopuffer_api_key_configured = api_key_configured(
        TURBOPUFFER_API_KEY_ENVS,
        explicit_key=tp.api_key,
        get_env=get_env_stripped,
    )
    region = tp.region or get_env_stripped(TURBOPUFFER_REGION_ENV) or None
    base_url_configured = bool(
        tp.base_url or get_env_stripped(TURBOPUFFER_BASE_URL_ENV)
    )
    return {
        FIELD_MATURITY: TURBOPUFFER_VECTOR_STORE_PROVIDER_SPEC.diagnostic_maturity,
        FIELD_CONFIGURED: (
            config.vector_store.provider == TURBOPUFFER_VECTOR_STORE_PROVIDER_SPEC.name
        ),
        FIELD_PACKAGE_AVAILABLE: package_available,
        FIELD_API_KEY_CONFIGURED: turbopuffer_api_key_configured,
        "credential_required": True,
        "runtime_validated": False,
        "runtime_validation": VECTOR_STORE_RUNTIME_NOT_REQUESTED,
        "check_store_supported": True,
        "extra": TURBOPUFFER_VECTOR_STORE_PROVIDER_SPEC.name,
        "namespace": tp.namespace,
        "region": region,
        "base_url_configured": base_url_configured,
        "distance_metric": tp.distance_metric,
        FIELD_RUNTIME_CONFIG: "Config.vector_store",
        "query_plan_scope": VECTOR_STORE_QUERY_PLAN_SCOPE_ADAPTER_MAXIMUM,
        **_capability_payload(TURBOPUFFER_VECTOR_STORE_PROVIDER_SPEC.capabilities),
    }


def _memory_diagnostics() -> dict[str, object]:
    return {
        FIELD_MATURITY: MEMORY_VECTOR_STORE_PROVIDER_SPEC.diagnostic_maturity,
        FIELD_CONFIGURED: False,
        FIELD_PACKAGE_AVAILABLE: True,
        "credential_required": False,
        "runtime_validated": False,
        "runtime_validation": VECTOR_STORE_RUNTIME_NOT_REQUESTED,
        "check_store_supported": False,
        FIELD_RUNTIME_CONFIG: "Engine(vector_store=InMemoryVectorStore(...))",
        "query_plan_scope": VECTOR_STORE_QUERY_PLAN_SCOPE_ADAPTER_MAXIMUM,
        **_capability_payload(MEMORY_VECTOR_STORE_PROVIDER_SPEC.capabilities),
    }


def _capability_payload(spec: VectorStoreCapabilitySpec) -> dict[str, object]:
    return {
        "per_point_delete": spec.per_point_delete,
        "document_record_lookup": spec.document_record_lookup,
        "query_plan": describe_query_plan_capabilities(spec.query_plan),
        "metadata_filter": describe_metadata_filter_capabilities(spec.metadata_filter),
    }
