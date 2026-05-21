from __future__ import annotations

import rag_core.integrations as integrations
from rag_core.integrations.langchain import (
    LangChainNotInstalledError,
    LangChainRetrieverConfig,
    build_langchain_retriever,
    create_langchain_context_tool,
    create_langchain_retriever_tool,
)
from rag_core.integrations.openai_agents import build_retrieve_context_tool


def test_integration_root_exports_stable_builders() -> None:
    assert integrations.__all__ == (
        "LangChainNotInstalledError",
        "LangChainRetrieverConfig",
        "build_langchain_retriever",
        "build_retrieve_context_tool",
        "create_langchain_context_tool",
        "create_langchain_retriever_tool",
        "langchain",
        "openai_agents",
    )
    assert integrations.LangChainNotInstalledError is LangChainNotInstalledError
    assert integrations.LangChainRetrieverConfig is LangChainRetrieverConfig
    assert integrations.build_langchain_retriever is build_langchain_retriever
    assert integrations.build_retrieve_context_tool is build_retrieve_context_tool
    assert integrations.create_langchain_context_tool is create_langchain_context_tool
    assert integrations.create_langchain_retriever_tool is create_langchain_retriever_tool


def test_integration_root_keeps_payload_helpers_in_submodules() -> None:
    assert not hasattr(integrations, "context_pack_to_tool_output")
    assert not hasattr(integrations, "search_result_to_document_kwargs")
