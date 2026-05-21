"""Retriever search helpers for the LangChain integration."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, cast
from rag_core.core import RAGCore
from rag_core.search.types import SearchResult


class LangChainSearchConfig(Protocol):
    @property
    def namespace(self) -> str: ...

    @property
    def corpus_ids(self) -> Sequence[str]: ...

    @property
    def limit(self) -> int: ...

    @property
    def document_ids(self) -> Sequence[str] | None: ...

    @property
    def rerank(self) -> bool: ...

    @property
    def use_lexical_search(self) -> bool: ...

    @property
    def query_plan(self) -> object | None: ...


async def search_langchain_documents(
    *,
    core: RAGCore,
    query: str,
    config: LangChainSearchConfig,
) -> list[SearchResult]:
    return await core.search(
        query=query,
        namespace=config.namespace,
        corpus_ids=list(config.corpus_ids),
        limit=config.limit,
        document_ids=(
            list(config.document_ids)
            if config.document_ids is not None
            else None
        ),
        rerank=config.rerank,
        use_lexical_search=config.use_lexical_search,
        query_plan=cast(Any, config.query_plan),
    )
