"""MCP's SDK validates input and preserves app-bound scope before retrieval."""

from __future__ import annotations
import asyncio
import json
from typing import Any
import pytest
from mcp import types
from rag_core.integrations.mcp_server import build_mcp_server
from rag_core.search.context_pack import build_context_pack
from tests.support import make_search_result


class Core:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    async def context(self, **kwargs):
        self.calls.append(kwargs)
        return build_context_pack(
            [make_search_result(text="Pay by ACH.")], query=kwargs["query"]
        )


@pytest.mark.integration
def test_mcp_sdk_calls_the_same_context_path_with_fixed_scope():
    async def run():
        core = Core()
        server = build_mcp_server(
            core, tenant_id="acme", collection="docs", limit_cap=3
        )
        request = types.CallToolRequest(
            params=types.CallToolRequestParams(
                name="search_user_documents",
                arguments={"query": "invoice payment", "limit": 2},
            )
        )
        response = await server.request_handlers[types.CallToolRequest](request)
        assert isinstance(response.root, types.CallToolResult)
        assert isinstance(response.root.content[0], types.TextContent)
        payload = json.loads(response.root.content[0].text)
        assert payload["ok"] is True and "ACH" in payload["context_text"]
        assert core.calls[0]["tenant_id"] == "acme"
        assert core.calls[0]["collections"] == ["docs"]
        assert core.calls[0]["limit"] == 2

    asyncio.run(run())


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": "q", "tenant_id": "other"},
        {"query": "q", "collection": "private"},
        {"query": "q", "limit": True},
        {"query": "q", "limit": 1000},
        {"query": " "},
    ],
)
def test_mcp_invalid_requests_do_not_retrieve(arguments):
    async def run():
        core = Core()
        server = build_mcp_server(core, tenant_id="acme", collection="docs")
        response = await server.request_handlers[types.CallToolRequest](
            types.CallToolRequest(
                params=types.CallToolRequestParams(
                    name="search_user_documents", arguments=arguments
                )
            )
        )
        assert isinstance(response.root, types.CallToolResult)
        assert response.root.isError is True
        assert core.calls == []

    asyncio.run(run())
