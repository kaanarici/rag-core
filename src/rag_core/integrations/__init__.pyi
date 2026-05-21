from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType
from typing import Any, Literal, Protocol

from rag_core.core import RAGCore
from rag_core.integrations.langchain import (
    LangChainNotInstalledError as LangChainNotInstalledError,
    LangChainRetrieverConfig as LangChainRetrieverConfig,
)

__all__: tuple[str, ...]
langchain: ModuleType
openai_agents: ModuleType

class SupportsRetrieveContext(Protocol):
    async def retrieve_context(
        self,
        *,
        query: str,
        namespace: str,
        corpus_ids: list[str],
        limit: int,
        document_ids: list[str] | None,
        rerank: bool,
        use_lexical_search: bool,
        max_chars: int | None,
        max_tokens: int | None,
    ) -> Any: ...

def build_langchain_retriever(
    core: RAGCore,
    *,
    namespace: str,
    corpus_ids: Sequence[str],
    limit: int = ...,
    document_ids: Sequence[str] | None = ...,
    rerank: bool = ...,
    use_lexical_search: bool = ...,
    query_plan: object | None = ...,
) -> Any: ...
def build_retrieve_context_tool(
    core: SupportsRetrieveContext,
    *,
    namespace: str,
    corpus_ids: Sequence[str],
    document_ids: Sequence[str] | None = ...,
    tool_name: str = ...,
    tool_description: str | None = ...,
    default_limit: int = ...,
    default_rerank: bool = ...,
    default_use_lexical_search: bool = ...,
    default_max_chars: int | None = ...,
    default_max_tokens: int | None = ...,
    return_payload: bool = ...,
    timeout: float | None = ...,
) -> Any: ...
def create_langchain_context_tool(
    core: RAGCore,
    *,
    namespace: str,
    corpus_ids: Sequence[str],
    name: str = ...,
    description: str = ...,
    limit: int = ...,
    document_ids: Sequence[str] | None = ...,
    rerank: bool = ...,
    use_lexical_search: bool = ...,
    max_chars: int | None = ...,
    max_tokens: int | None = ...,
    query_plan: object | None = ...,
) -> Any: ...
def create_langchain_retriever_tool(
    retriever: Any,
    *,
    name: str,
    description: str,
    document_separator: str = ...,
    response_format: Literal["content", "content_and_artifact"] = ...,
) -> Any: ...
