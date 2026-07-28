#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

uv sync --group dev --extra langchain --extra openai-agents --extra mcp

cd "$REPO_ROOT"
uv run pytest -q \
  tests/test_langchain_integration.py \
  tests/test_langchain_context_tool_contract.py \
  tests/test_mcp_server.py \
  tests/test_openai_agents_tool.py \
  tests/test_openai_agents_request_contract.py

uv run python - <<'PY'
from rag_core import Config, RAGCore
from rag_core.demo import build_demo_core
from rag_core.integrations.langchain import create_langchain_retriever_tool
from rag_core.integrations.mcp_server import build_mcp_server
from rag_core.search.providers import create_reranker
from mcp import types

class _Retriever:
    async def ainvoke(self, query):
        del query
        return []

    def invoke(self, query):
        del query
        return []

tool = create_langchain_retriever_tool(
    _Retriever(),
    name="rag_lookup",
    description="Look up app-owned documents.",
)
if getattr(tool, "name", None) != "rag_lookup":
    raise AssertionError("LangChain retriever tool did not expose the expected name")

class _Core:
    async def search(self, **kwargs):
        del kwargs
        raise AssertionError("MCP smoke should not execute search")

    async def context(self, **kwargs):
        del kwargs
        raise AssertionError("OpenAI Agents smoke should not execute retrieval")

rag = RAGCore(
    build_demo_core(store_collection="optional_integrations"),
    tenant_id="acme",
    index="help",
)
agents_tool = rag.tool()
if getattr(agents_tool, "name", None) != "search_user_documents":
    raise AssertionError("RAGCore.tool() did not expose the expected name")

mcp_server = build_mcp_server(
    _Core(),
    namespace="acme",
    collections=["help"],
)

async def list_mcp_tools():
    result = await mcp_server.request_handlers[types.ListToolsRequest](
        types.ListToolsRequest()
    )
    return result.root.tools

import asyncio

mcp_tools = asyncio.run(list_mcp_tools())
asyncio.run(rag.close())
if [tool.name for tool in mcp_tools] != ["search_user_documents"]:
    raise AssertionError("MCP server did not expose the expected tools")

reranker = create_reranker(provider="none")
if reranker.provider_name != Config().reranker.provider:
    raise AssertionError("default reranker construction failed")
PY

echo "optional integration smoke passed"
