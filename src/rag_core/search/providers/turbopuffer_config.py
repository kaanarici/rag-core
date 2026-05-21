from __future__ import annotations

import re
from dataclasses import dataclass

from rag_core.search.policy import DEFAULT_POLICY, VectorStorePolicy

_SUPPORTED_DISTANCE_METRICS = frozenset({"cosine_distance", "euclidean_squared"})
_NAMESPACE_RE = re.compile(r"^[A-Za-z0-9\-_.]{1,128}$")
DEFAULT_TURBOPUFFER_WRITE_BATCH_SIZE = 1_000
DEFAULT_TURBOPUFFER_DELETE_CONTINUATION_LIMIT = 1_000


@dataclass(frozen=True)
class TurboPufferConfig:
    namespace: str
    dense_dimensions: int
    region: str | None
    base_url: str | None
    distance_metric: str
    write_batch_size: int
    delete_continuation_limit: int
    policy: VectorStorePolicy = DEFAULT_POLICY


def build_turbopuffer_config(
    *,
    namespace: str,
    dense_dimensions: int,
    region: str | None,
    base_url: str | None,
    distance_metric: str,
    write_batch_size: int,
    delete_continuation_limit: int,
    policy: VectorStorePolicy,
) -> TurboPufferConfig:
    resolved_namespace = namespace.strip()
    if not resolved_namespace:
        raise ValueError("namespace is required for TurboPufferVectorStore")
    if not _NAMESPACE_RE.fullmatch(resolved_namespace):
        raise ValueError(
            "TurboPufferVectorStore namespace must match [A-Za-z0-9-_.]{1,128}"
        )
    if dense_dimensions <= 0:
        raise ValueError("dense_dimensions must be positive")
    if distance_metric not in _SUPPORTED_DISTANCE_METRICS:
        supported = ", ".join(sorted(_SUPPORTED_DISTANCE_METRICS))
        raise ValueError(
            f"unsupported TurboPuffer distance_metric {distance_metric!r}; "
            f"choose one of: {supported}"
        )
    resolved_write_batch_size = validate_turbopuffer_write_batch_size(write_batch_size)
    resolved_delete_continuation_limit = validate_turbopuffer_delete_continuation_limit(
        delete_continuation_limit
    )
    return TurboPufferConfig(
        namespace=resolved_namespace,
        dense_dimensions=dense_dimensions,
        region=region,
        base_url=base_url,
        distance_metric=distance_metric,
        write_batch_size=resolved_write_batch_size,
        delete_continuation_limit=resolved_delete_continuation_limit,
        policy=policy,
    )


def validate_turbopuffer_write_batch_size(write_batch_size: int) -> int:
    if (
        isinstance(write_batch_size, bool)
        or not isinstance(write_batch_size, int)
        or write_batch_size <= 0
    ):
        raise ValueError("TurboPufferVectorStore write_batch_size must be positive")
    return write_batch_size


def validate_turbopuffer_delete_continuation_limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError(
            "TurboPufferVectorStore delete_continuation_limit must be positive"
        )
    return limit


def owns_turbopuffer_client(
    *,
    client: object | None,
    namespace_client: object | None,
) -> bool:
    if client is not None and namespace_client is not None:
        raise ValueError("provide client or namespace_client, not both")
    return client is None and namespace_client is None
