from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType
from typing import Any, Literal

import rag_core.integrations.protocols
from rag_core import RAGCore
from rag_core.events.types import AuditContext
from rag_core.integrations.langchain import (
    LangChainNotInstalledError as LangChainNotInstalledError,
    LangChainRetrieverConfig as LangChainRetrieverConfig,
)
from rag_core.search import QueryPlan
from rag_core.search.context_pack import ContextOrder
from mcp.server import Server

__all__: tuple[str, ...] = (
    "LangChainNotInstalledError",
    "LangChainRetrieverConfig",
    "build_langchain_retriever",
    "build_mcp_server",
    "build_retrieve_context_tool",
    "create_langchain_context_tool",
    "create_langchain_retriever_tool",
    "langchain",
    "mcp_server",
    "openai_agents",
)
langchain: ModuleType
mcp_server: ModuleType
openai_agents: ModuleType

def build_langchain_retriever(
    core: RAGCore,
    *,
    namespace: str,
    corpus_ids: Sequence[str],
    limit: int = ...,
    content_types: Sequence[str] | None = ...,
    document_ids: Sequence[str] | None = ...,
    rerank: bool = ...,
    use_lexical_search: bool = ...,
    query_plan: QueryPlan | None = ...,
    audit_context: AuditContext | None = ...,
) -> Any: ...
def build_retrieve_context_tool(
    core: rag_core.integrations.protocols.SupportsRetrieveContext,
    *,
    namespace: str,
    corpus_ids: Sequence[str],
    content_types: Sequence[str] | None = ...,
    document_ids: Sequence[str] | None = ...,
    tool_name: str = ...,
    tool_description: str | None = ...,
    default_limit: int = ...,
    default_rerank: bool = ...,
    default_use_lexical_search: bool = ...,
    default_max_chars: int | None = ...,
    default_max_tokens: int | None = ...,
    query_plan: QueryPlan | None = ...,
    context_order: ContextOrder = ...,
    audit_context: AuditContext | None = ...,
    return_payload: bool = ...,
    timeout: float | None = ...,
) -> Any: ...
def build_mcp_server(
    core: rag_core.integrations.protocols.SupportsRetrieveContext,
    *,
    namespace: str,
    corpus_ids: Sequence[str],
    rerank: bool = ...,
    limit_cap: int = ...,
    context_order: ContextOrder = ...,
    server_name: str = ...,
) -> Server[object, object]: ...
def create_langchain_context_tool(
    core: RAGCore,
    *,
    namespace: str,
    corpus_ids: Sequence[str],
    name: str = ...,
    description: str = ...,
    limit: int = ...,
    content_types: Sequence[str] | None = ...,
    document_ids: Sequence[str] | None = ...,
    rerank: bool = ...,
    use_lexical_search: bool = ...,
    max_chars: int | None = ...,
    max_tokens: int | None = ...,
    query_plan: QueryPlan | None = ...,
    context_order: ContextOrder = ...,
    audit_context: AuditContext | None = ...,
) -> Any: ...
def create_langchain_retriever_tool(
    retriever: Any,
    *,
    name: str,
    description: str,
    document_separator: str = ...,
    response_format: Literal["content", "content_and_artifact"] = ...,
) -> Any: ...
