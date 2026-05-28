from __future__ import annotations

import re
from dataclasses import dataclass, field

QDRANT_VECTOR_STORE_PROVIDER = "qdrant"
TURBOPUFFER_VECTOR_STORE_PROVIDER = "turbopuffer"
VECTOR_STORE_ENV = "RAG_CORE_VECTOR_STORE"
DEFAULT_VECTOR_STORE_PROVIDER = QDRANT_VECTOR_STORE_PROVIDER
SUPPORTED_VECTOR_STORE_PROVIDERS = (
    QDRANT_VECTOR_STORE_PROVIDER,
    TURBOPUFFER_VECTOR_STORE_PROVIDER,
)
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
