"""Optional integration adapters for external ecosystems."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = (
    "LangChainNotInstalledError",
    "LangChainRetrieverConfig",
    "build_langchain_retriever",
    "build_retrieve_context_tool",
    "create_langchain_context_tool",
    "create_langchain_retriever_tool",
    "langchain",
    "openai_agents",
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "LangChainNotInstalledError": (
        "rag_core.integrations.langchain",
        "LangChainNotInstalledError",
    ),
    "LangChainRetrieverConfig": (
        "rag_core.integrations.langchain",
        "LangChainRetrieverConfig",
    ),
    "build_langchain_retriever": (
        "rag_core.integrations.langchain",
        "build_langchain_retriever",
    ),
    "build_retrieve_context_tool": (
        "rag_core.integrations.openai_agents",
        "build_retrieve_context_tool",
    ),
    "create_langchain_context_tool": (
        "rag_core.integrations.langchain",
        "create_langchain_context_tool",
    ),
    "create_langchain_retriever_tool": (
        "rag_core.integrations.langchain",
        "create_langchain_retriever_tool",
    ),
}


def __getattr__(name: str) -> Any:
    if name in ("langchain", "openai_agents"):
        return import_module(f"rag_core.integrations.{name}")
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, symbol = target
    return getattr(import_module(module_name), symbol)


def __dir__() -> list[str]:
    return list(__all__)
