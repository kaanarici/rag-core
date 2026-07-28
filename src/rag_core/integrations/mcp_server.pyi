from __future__ import annotations

from collections.abc import Sequence

from mcp.server import Server

from rag_core.integrations.protocols import SupportsRetrieveContext
from rag_core.search.context_pack import ContextOrder

def build_mcp_server(
    core: SupportsRetrieveContext,
    *,
    collection: str | None = ...,
    collections: Sequence[str] | None = ...,
    namespace: str | None = ...,
    rerank: bool = ...,
    limit_cap: int = ...,
    context_order: ContextOrder = ...,
    server_name: str = ...,
) -> Server[object, object]: ...

__all__: tuple[str, ...]
