from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from rag_core import Engine
from rag_core.contracts import SupportsContextPromptPayload
from rag_core.events.types import AuditContext
from rag_core.integrations.langchain_runtime import (
    LangChainNotInstalledError as LangChainNotInstalledError,
)
from rag_core.search import QueryPlan, SearchResult
from rag_core.search.context_pack import ContextOrder

@dataclass(frozen=True)
class LangChainRetrieverConfig:
    namespace: str
    collections: tuple[str, ...]
    limit: int = ...
    content_types: tuple[str, ...] | None = ...
    document_ids: tuple[str, ...] | None = ...
    rerank: bool = ...
    use_lexical_search: bool = ...
    query_plan: QueryPlan | None = ...
    audit_context: AuditContext | None = ...

def build_langchain_retriever(
    core: Engine,
    *,
    collection: str | None = ...,
    collections: Sequence[str] | None = ...,
    namespace: str | None = ...,
    limit: int = ...,
    content_types: Sequence[str] | None = ...,
    document_ids: Sequence[str] | None = ...,
    rerank: bool = ...,
    use_lexical_search: bool = ...,
    query_plan: QueryPlan | None = ...,
    audit_context: AuditContext | None = ...,
) -> Any: ...
def context_pack_to_tool_output(
    pack: SupportsContextPromptPayload,
    *,
    context_order: ContextOrder = ...,
) -> tuple[str, dict[str, object]]: ...
def create_langchain_context_tool(
    core: Engine,
    *,
    collection: str | None = ...,
    collections: Sequence[str] | None = ...,
    namespace: str | None = ...,
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
def search_result_to_document_kwargs(result: SearchResult) -> dict[str, object]: ...

__all__: list[str] = [
    "LangChainNotInstalledError",
    "LangChainRetrieverConfig",
    "build_langchain_retriever",
    "context_pack_to_tool_output",
    "create_langchain_context_tool",
    "create_langchain_retriever_tool",
    "search_result_to_document_kwargs",
]
