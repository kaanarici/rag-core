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
from rag_core import RAGCoreConfig
from rag_core.integrations.langchain import create_langchain_retriever_tool
from rag_core.integrations.mcp_server import build_mcp_server
from rag_core.integrations.openai_agents import build_retrieve_context_tool
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

    async def retrieve_context(self, **kwargs):
        del kwargs
        raise AssertionError("OpenAI Agents smoke should not execute retrieval")

agents_tool = build_retrieve_context_tool(
    _Core(),
    namespace="acme",
    corpus_ids=["help"],
    tool_name="search_user_documents",
)
if getattr(agents_tool, "name", None) != "search_user_documents":
    raise AssertionError("OpenAI Agents tool did not expose the expected name")

mcp_server = build_mcp_server(
    _Core(),
    namespace="acme",
    corpus_ids=["help"],
)

async def list_mcp_tools():
    result = await mcp_server.request_handlers[types.ListToolsRequest](
        types.ListToolsRequest()
    )
    return result.root.tools

import asyncio

mcp_tools = asyncio.run(list_mcp_tools())
if [tool.name for tool in mcp_tools] != ["search_user_documents"]:
    raise AssertionError("MCP server did not expose the expected tools")

reranker = create_reranker(provider="none")
if reranker.provider_name != RAGCoreConfig().reranker.provider:
    raise AssertionError("default reranker construction failed")
PY

echo "optional integration smoke passed"
