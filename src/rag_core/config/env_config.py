from __future__ import annotations

from rag_core.config.embedding_config import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEMO_EMBEDDING_MODEL,
    DEMO_EMBEDDING_PROVIDER,
    EMBEDDING_BATCH_SIZE_ENV,
    EMBEDDING_DIMENSIONS_ENV,
    EMBEDDING_MODEL_ENV,
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_PROVIDER,
    EmbeddingConfig,
)
from rag_core.config.env_access import (
    get_env_bool_strict,
    get_env_int_strict,
    get_env_optional,
    get_env_stripped,
)
from rag_core.config.ingest_config import (
    DEFAULT_PROCESSING_VERSION,
    PROCESSING_VERSION_ENV,
    IngestConfig,
)
from rag_core.config.qdrant_config import (
    DEFAULT_QDRANT_COLLECTION,
    DEFAULT_QDRANT_DIMENSION_AWARE_COLLECTION,
    QDRANT_COLLECTION_ENV,
    QDRANT_DIMENSION_AWARE_COLLECTION_ENV,
    QDRANT_LOCATION_ENV,
    QDRANT_URL_ENV,
    QdrantConfig,
)
from rag_core.config.vector_store_config import (
    DEFAULT_PGVECTOR_SCHEMA,
    DEFAULT_TURBOPUFFER_DELETE_CONTINUATION_LIMIT,
    DEFAULT_TURBOPUFFER_DISTANCE_METRIC,
    DEFAULT_VECTOR_STORE_PROVIDER,
    PGVECTOR_DSN_ENV,
    PGVECTOR_SCHEMA_ENV,
    PGVECTOR_TABLE_ENV,
    PGVECTOR_VECTOR_STORE_PROVIDER,
    QDRANT_VECTOR_STORE_PROVIDER,
    TURBOPUFFER_BASE_URL_ENV,
    TURBOPUFFER_DELETE_CONTINUATION_LIMIT_ENV,
    TURBOPUFFER_DISTANCE_METRIC_ENV,
    TURBOPUFFER_NAMESPACE_ENV,
    TURBOPUFFER_REGION_ENV,
    TURBOPUFFER_VECTOR_STORE_PROVIDER,
    VECTOR_STORE_ENV,
    PgVectorStoreConfig,
    TurboPufferVectorStoreConfig,
    VectorStoreConfig,
)
from rag_core.core_models import Config
from rag_core.provider_api_keys import (
    QDRANT_API_KEY_ENVS,
    TURBOPUFFER_API_KEY_ENVS,
    first_configured_api_key,
    provider_api_key_env_names,
)
from rag_core.search.providers.cohere import (
    COHERE_PROVIDER,
    DEFAULT_COHERE_EMBEDDING_MODEL,
)
from rag_core.search.providers.voyage import (
    DEFAULT_VOYAGE_EMBEDDING_MODEL,
    VOYAGE_PROVIDER,
)
from rag_core.search.providers.zeroentropy import (
    DEFAULT_ZEROENTROPY_EMBEDDING_MODEL,
    ZEROENTROPY_PROVIDER,
)


def build_config_from_env() -> Config:
    """Build one strict production configuration without loading dotenv files."""
    embedding_provider = get_env_stripped(
        EMBEDDING_PROVIDER_ENV,
        DEFAULT_EMBEDDING_PROVIDER,
    ).lower()
    embedding_model = _optional_env(EMBEDDING_MODEL_ENV)
    if embedding_model is None:
        embedding_model = _default_embedding_model(embedding_provider)
    vector_store_provider = get_env_stripped(
        VECTOR_STORE_ENV,
        DEFAULT_VECTOR_STORE_PROVIDER,
    ).lower()

    qdrant_url = _optional_env(QDRANT_URL_ENV)
    qdrant_location = _optional_env(QDRANT_LOCATION_ENV)
    if vector_store_provider == QDRANT_VECTOR_STORE_PROVIDER:
        if bool(qdrant_url) == bool(qdrant_location):
            raise ValueError(
                f"{VECTOR_STORE_ENV}=qdrant requires exactly one of "
                f"{QDRANT_URL_ENV} or {QDRANT_LOCATION_ENV}"
            )
    else:
        qdrant_url = None
        qdrant_location = None

    pgvector_dsn = (
        _optional_env(PGVECTOR_DSN_ENV)
        if vector_store_provider == PGVECTOR_VECTOR_STORE_PROVIDER
        else None
    )
    turbopuffer_namespace = (
        _optional_env(TURBOPUFFER_NAMESPACE_ENV)
        if vector_store_provider == TURBOPUFFER_VECTOR_STORE_PROVIDER
        else None
    )
    if (
        vector_store_provider == TURBOPUFFER_VECTOR_STORE_PROVIDER
        and turbopuffer_namespace is None
    ):
        raise ValueError(
            f"{VECTOR_STORE_ENV}=turbopuffer requires {TURBOPUFFER_NAMESPACE_ENV}"
        )

    return Config(
        vector_store=VectorStoreConfig(
            provider=vector_store_provider,
            qdrant=QdrantConfig(
                url=qdrant_url,
                location=qdrant_location,
                api_key=(
                    first_configured_api_key(QDRANT_API_KEY_ENVS) or None
                    if vector_store_provider == QDRANT_VECTOR_STORE_PROVIDER
                    else None
                ),
                store_collection=get_env_stripped(
                    QDRANT_COLLECTION_ENV,
                    DEFAULT_QDRANT_COLLECTION,
                ),
                dimension_aware_collection=get_env_bool_strict(
                    QDRANT_DIMENSION_AWARE_COLLECTION_ENV,
                    DEFAULT_QDRANT_DIMENSION_AWARE_COLLECTION,
                ),
            ),
            pgvector=PgVectorStoreConfig(
                dsn=pgvector_dsn,
                schema=get_env_stripped(PGVECTOR_SCHEMA_ENV, DEFAULT_PGVECTOR_SCHEMA),
                table=(
                    _optional_env(PGVECTOR_TABLE_ENV)
                    if vector_store_provider == PGVECTOR_VECTOR_STORE_PROVIDER
                    else None
                ),
            ),
            turbopuffer=TurboPufferVectorStoreConfig(
                namespace=turbopuffer_namespace,
                api_key=(
                    first_configured_api_key(TURBOPUFFER_API_KEY_ENVS) or None
                    if vector_store_provider == TURBOPUFFER_VECTOR_STORE_PROVIDER
                    else None
                ),
                region=(
                    _optional_env(TURBOPUFFER_REGION_ENV)
                    if vector_store_provider == TURBOPUFFER_VECTOR_STORE_PROVIDER
                    else None
                ),
                base_url=(
                    _optional_env(TURBOPUFFER_BASE_URL_ENV)
                    if vector_store_provider == TURBOPUFFER_VECTOR_STORE_PROVIDER
                    else None
                ),
                distance_metric=get_env_stripped(
                    TURBOPUFFER_DISTANCE_METRIC_ENV,
                    DEFAULT_TURBOPUFFER_DISTANCE_METRIC,
                ),
                delete_continuation_limit=get_env_int_strict(
                    TURBOPUFFER_DELETE_CONTINUATION_LIMIT_ENV,
                    DEFAULT_TURBOPUFFER_DELETE_CONTINUATION_LIMIT,
                ),
            ),
        ),
        embedding=EmbeddingConfig(
            provider=embedding_provider,
            model=embedding_model,
            dimensions=_optional_int_env(EMBEDDING_DIMENSIONS_ENV),
            api_key=(
                first_configured_api_key(
                    provider_api_key_env_names(embedding_provider)
                )
                or None
            ),
            batch_size=get_env_int_strict(
                EMBEDDING_BATCH_SIZE_ENV,
                DEFAULT_EMBEDDING_BATCH_SIZE,
            ),
        ),
        ingest=IngestConfig(
            processing_version=get_env_stripped(
                PROCESSING_VERSION_ENV,
                DEFAULT_PROCESSING_VERSION,
            ),
        ),
    )


def _default_embedding_model(provider: str) -> str:
    models = {
        DEFAULT_EMBEDDING_PROVIDER: DEFAULT_EMBEDDING_MODEL,
        DEMO_EMBEDDING_PROVIDER: DEMO_EMBEDDING_MODEL,
        LOCAL_EMBEDDING_PROVIDER: LOCAL_EMBEDDING_MODEL,
        COHERE_PROVIDER: DEFAULT_COHERE_EMBEDDING_MODEL,
        VOYAGE_PROVIDER: DEFAULT_VOYAGE_EMBEDDING_MODEL,
        ZEROENTROPY_PROVIDER: DEFAULT_ZEROENTROPY_EMBEDDING_MODEL,
    }
    try:
        return models[provider]
    except KeyError as exc:
        raise ValueError(
            f"{EMBEDDING_MODEL_ENV} is required for embedding provider "
            f"{provider!r}"
        ) from exc


def _optional_env(name: str) -> str | None:
    value = get_env_optional(name)
    if value is None:
        return None
    return value.strip() or None


def _optional_int_env(name: str) -> int | None:
    value = _optional_env(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


__all__ = ["build_config_from_env"]
