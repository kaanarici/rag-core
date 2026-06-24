"""TurboPuffer client, config, and health helpers."""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import Protocol, cast

from rag_core.config.vector_store_config import SUPPORTED_TURBOPUFFER_DISTANCE_METRICS
from rag_core.search.policy import DEFAULT_POLICY, VectorStorePolicy

from .vector_store_capabilities import TURBOPUFFER_VECTOR_STORE_PROVIDER_SPEC

_NAMESPACE_RE = re.compile(r"^[A-Za-z0-9\-_.]{1,128}$")
DEFAULT_TURBOPUFFER_WRITE_BATCH_SIZE = 1_000


class TurboPufferNamespace(Protocol):
    async def metadata(self) -> object: ...

    async def write(self, **kwargs: object) -> object: ...

    async def query(self, **kwargs: object) -> object: ...


class _TurboPufferClient(Protocol):
    def namespace(self, name: str) -> TurboPufferNamespace: ...


@dataclass(frozen=True)
class TurboPufferNamespaceState:
    namespace: TurboPufferNamespace
    client: object | None


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
    if distance_metric not in SUPPORTED_TURBOPUFFER_DISTANCE_METRICS:
        supported = ", ".join(sorted(SUPPORTED_TURBOPUFFER_DISTANCE_METRICS))
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


async def close_turbopuffer_client(*, owns_client: bool, client: object | None) -> None:
    if not owns_client or client is None:
        return
    close = getattr(client, "close", None) or getattr(client, "aclose", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result


def resolve_turbopuffer_namespace(
    *,
    namespace_client: object | None,
    client: object | None,
    namespace: str,
    api_key: str | None,
    region: str | None,
    base_url: str | None,
) -> TurboPufferNamespaceState:
    if namespace_client is not None:
        return TurboPufferNamespaceState(
            namespace=cast(TurboPufferNamespace, namespace_client),
            client=client,
        )

    resolved_client = client
    if resolved_client is None:
        try:
            from turbopuffer import AsyncTurbopuffer
        except ImportError as exc:
            raise ImportError(
                "turbopuffer package is required for TurboPufferVectorStore. "
                "Install rag-core[turbopuffer] or provide namespace_client."
            ) from exc

        resolved_client = AsyncTurbopuffer(
            api_key=api_key,
            region=region,
            base_url=base_url,
        )

    turbopuffer_client = cast(_TurboPufferClient, resolved_client)
    return TurboPufferNamespaceState(
        namespace=turbopuffer_client.namespace(namespace),
        client=resolved_client,
    )


def _build_healthy_health(*, namespace: str, metadata: object) -> dict[str, object]:
    index = getattr(metadata, "index", None)
    return {
        "healthy": True,
        "adapter": TURBOPUFFER_VECTOR_STORE_PROVIDER_SPEC.name,
        "namespace": namespace,
        "points_count": getattr(metadata, "approx_row_count", None),
        "logical_bytes": getattr(metadata, "approx_logical_bytes", None),
        "index_status": getattr(index, "status", None),
    }


def _build_unhealthy_health(*, namespace: str, exc: Exception) -> dict[str, object]:
    return {
        "healthy": False,
        "adapter": TURBOPUFFER_VECTOR_STORE_PROVIDER_SPEC.name,
        "namespace": namespace,
        "error": type(exc).__name__,
    }
