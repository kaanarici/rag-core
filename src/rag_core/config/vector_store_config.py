from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag_core.fetch_security import FetchSecurityPolicy, validate_fetch_url

QDRANT_VECTOR_STORE_PROVIDER = "qdrant"
PGVECTOR_VECTOR_STORE_PROVIDER = "pgvector"
TURBOPUFFER_VECTOR_STORE_PROVIDER = "turbopuffer"
VECTOR_STORE_ENV = "RAG_CORE_VECTOR_STORE"
DEFAULT_VECTOR_STORE_PROVIDER = QDRANT_VECTOR_STORE_PROVIDER
SUPPORTED_VECTOR_STORE_PROVIDERS = (
    QDRANT_VECTOR_STORE_PROVIDER,
    PGVECTOR_VECTOR_STORE_PROVIDER,
    TURBOPUFFER_VECTOR_STORE_PROVIDER,
)
PGVECTOR_DSN_ENV = "RAG_CORE_PGVECTOR_DSN"
PGVECTOR_SCHEMA_ENV = "RAG_CORE_PGVECTOR_SCHEMA"
PGVECTOR_TABLE_ENV = "RAG_CORE_PGVECTOR_TABLE"
DEFAULT_PGVECTOR_SCHEMA = "public"
SUPPORTED_TURBOPUFFER_DISTANCE_METRICS = ("cosine_distance", "euclidean_squared")
TURBOPUFFER_BASE_URL_ENV = "TURBOPUFFER_BASE_URL"
TURBOPUFFER_DELETE_CONTINUATION_LIMIT_ENV = (
    "RAG_CORE_TURBOPUFFER_DELETE_CONTINUATION_LIMIT"
)
TURBOPUFFER_DISTANCE_METRIC_ENV = "RAG_CORE_TURBOPUFFER_DISTANCE_METRIC"
TURBOPUFFER_NAMESPACE_ENV = "RAG_CORE_TURBOPUFFER_NAMESPACE"
TURBOPUFFER_REGION_ENV = "TURBOPUFFER_REGION"
DEFAULT_TURBOPUFFER_DISTANCE_METRIC = "cosine_distance"
DEFAULT_TURBOPUFFER_DELETE_CONTINUATION_LIMIT = 1_000
_TURBOPUFFER_NAMESPACE_RE = re.compile(r"^[A-Za-z0-9\-_.]{1,128}$")
_PGVECTOR_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")


@dataclass(frozen=True)
class PgVectorStoreConfig:
    dsn: str | None = None
    table: str | None = None
    schema: str = DEFAULT_PGVECTOR_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dsn",
            _optional_non_empty(self.dsn, "PgVectorStoreConfig.dsn"),
        )
        object.__setattr__(
            self,
            "table",
            _optional_pgvector_identifier(self.table, "PgVectorStoreConfig.table"),
        )
        object.__setattr__(
            self,
            "schema",
            _required_pgvector_identifier(self.schema, "PgVectorStoreConfig.schema"),
        )


@dataclass(frozen=True)
class TurboPufferVectorStoreConfig:
    namespace: str | None = None
    api_key: str | None = None
    region: str | None = None
    base_url: str | None = None
    distance_metric: str = DEFAULT_TURBOPUFFER_DISTANCE_METRIC
    delete_continuation_limit: int = DEFAULT_TURBOPUFFER_DELETE_CONTINUATION_LIMIT

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "namespace",
            _optional_turbopuffer_namespace(
                self.namespace, "TurboPufferVectorStoreConfig.namespace"
            ),
        )
        object.__setattr__(
            self,
            "api_key",
            _optional_non_empty(self.api_key, "TurboPufferVectorStoreConfig.api_key"),
        )
        object.__setattr__(
            self,
            "region",
            _optional_non_empty(self.region, "TurboPufferVectorStoreConfig.region"),
        )
        object.__setattr__(
            self,
            "base_url",
            _optional_non_empty(
                self.base_url, "TurboPufferVectorStoreConfig.base_url"
            ),
        )
        if self.base_url is not None:
            # Reject http://, embedded credentials, and (without explicit
            # opt-in) private-IP literals at config-time. The region pin then
            # asserts host-substring agreement. The assembly seam refuses to
            # construct a store whose resolved endpoint contradicts the pin.
            try:
                validated_base = validate_fetch_url(
                    self.base_url, policy=FetchSecurityPolicy()
                )
            except ValueError as exc:
                raise ValueError(
                    "TurboPufferVectorStoreConfig.base_url is not a safe https URL: "
                    f"{exc}"
                ) from None
            if self.region is not None and self.region not in validated_base.host:
                raise ValueError(
                    "TurboPufferVectorStoreConfig.base_url host does not match "
                    f"region pin {self.region!r}"
                )
        if not isinstance(self.distance_metric, str):
            raise ValueError(
                "TurboPufferVectorStoreConfig.distance_metric must be a non-empty string"
            )
        metric = self.distance_metric.strip()
        if not metric:
            raise ValueError(
                "TurboPufferVectorStoreConfig.distance_metric must be a non-empty string"
            )
        if metric not in SUPPORTED_TURBOPUFFER_DISTANCE_METRICS:
            known = ", ".join(SUPPORTED_TURBOPUFFER_DISTANCE_METRICS)
            raise ValueError(
                "TurboPufferVectorStoreConfig.distance_metric must be one of: "
                f"{known}"
            )
        object.__setattr__(self, "distance_metric", metric)
        if (
            isinstance(self.delete_continuation_limit, bool)
            or not isinstance(self.delete_continuation_limit, int)
            or self.delete_continuation_limit <= 0
        ):
            raise ValueError(
                "TurboPufferVectorStoreConfig.delete_continuation_limit "
                "must be positive"
            )


@dataclass(frozen=True)
class VectorStoreConfig:
    provider: str = DEFAULT_VECTOR_STORE_PROVIDER
    pgvector: PgVectorStoreConfig = field(default_factory=PgVectorStoreConfig)
    turbopuffer: TurboPufferVectorStoreConfig = field(
        default_factory=TurboPufferVectorStoreConfig
    )

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("VectorStoreConfig.provider must be a non-empty string")
        provider = self.provider.strip().lower()
        if provider not in SUPPORTED_VECTOR_STORE_PROVIDERS:
            known = ", ".join(SUPPORTED_VECTOR_STORE_PROVIDERS)
            raise ValueError(
                f"VectorStoreConfig.provider must be one of: {known}"
            )
        object.__setattr__(self, "provider", provider)
        if provider == PGVECTOR_VECTOR_STORE_PROVIDER and self.pgvector.dsn is None:
            raise ValueError(
                "VectorStoreConfig.pgvector.dsn is required when provider is pgvector"
            )


def _optional_non_empty(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    stripped = value.strip()
    return stripped or None


def _optional_turbopuffer_namespace(value: object, field_name: str) -> str | None:
    namespace = _optional_non_empty(value, field_name)
    if namespace is None:
        return None
    if not _TURBOPUFFER_NAMESPACE_RE.fullmatch(namespace):
        raise ValueError(f"{field_name} must match [A-Za-z0-9-_.]{{1,128}}")
    return namespace


def _optional_pgvector_identifier(value: object, field_name: str) -> str | None:
    identifier = _optional_non_empty(value, field_name)
    if identifier is None:
        return None
    return _validate_pgvector_identifier(identifier, field_name)


def _required_pgvector_identifier(value: object, field_name: str) -> str:
    identifier = _optional_non_empty(value, field_name)
    if identifier is None:
        raise ValueError(f"{field_name} must be a non-empty PostgreSQL identifier")
    return _validate_pgvector_identifier(identifier, field_name)


def _validate_pgvector_identifier(identifier: str, field_name: str) -> str:
    if not _PGVECTOR_IDENTIFIER_RE.fullmatch(identifier):
        raise ValueError(
            f"{field_name} must match [A-Za-z_][A-Za-z0-9_]{{0,62}}"
        )
    return identifier
