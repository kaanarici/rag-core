"""TurboPuffer client and namespace lifecycle helpers."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Protocol, cast


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
