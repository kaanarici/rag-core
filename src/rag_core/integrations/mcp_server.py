"""A thin MCP tool over the engine's cited context API."""

from __future__ import annotations
import json
from collections.abc import Sequence
from mcp import types
from mcp.server import Server
from rag_core.contracts import (
    normalize_static_retrieval_scope,
    validate_bound_tenant_id,
)
from rag_core.integrations.protocols import SupportsRetrieveContext
from rag_core.search.context_pack import ContextOrder


def build_mcp_server(
    core: SupportsRetrieveContext,
    *,
    collection: str | None = None,
    collections: Sequence[str] | None = None,
    tenant_id: str | None = None,
    rerank: bool = False,
    limit_cap: int = 10,
    context_order: ContextOrder = "rank",
    server_name: str = "rag-core",
) -> Server:
    tenant = validate_bound_tenant_id(tenant_id or "")
    scope, _ = normalize_static_retrieval_scope(
        collection=collection,
        collections=collections,
        document_ids=None,
        limit=limit_cap,
    )
    if isinstance(limit_cap, bool) or limit_cap <= 0:
        raise ValueError("limit_cap must be a positive integer")
    default_limit = min(5, limit_cap)
    server = Server(server_name)
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": limit_cap,
                "default": default_limit,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="search_user_documents",
                description="Search app-owned documents and return cited context.",
                inputSchema=schema,
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict | None) -> types.CallToolResult:
        try:
            if name != "search_user_documents":
                raise ValueError("unknown tool")
            args = arguments or {}
            if set(args) - {"query", "limit"}:
                raise ValueError("only query and limit may be supplied")
            query = args.get("query")
            limit = args.get("limit", default_limit)
            if not isinstance(query, str) or not query.strip():
                raise ValueError("query must be non-empty")
            if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
                raise ValueError("limit must be a positive integer")
            pack = await core.context(
                query=query,
                tenant_id=tenant,
                collections=list(scope),
                document_ids=None,
                content_types=None,
                limit=min(limit, limit_cap),
                rerank=rerank,
                max_chars=None,
                max_tokens=None,
                audit_context=None,
            )
            payload = {
                **pack.to_prompt_payload(),
                "ok": True,
                "context_text": pack.as_prompt_text(context_order=context_order),
            }
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text", text=json.dumps(payload, allow_nan=False)
                    )
                ]
            )
        except Exception as exc:
            return types.CallToolResult(
                isError=True,
                content=[
                    types.TextContent(
                        type="text",
                        text=json.dumps({"ok": False, "error": type(exc).__name__}),
                    )
                ],
            )

    return server
