from __future__ import annotations

from typing import TYPE_CHECKING

from rag_core.search.context_pack import ModelContextPack
from rag_core.search.types import RerankBudget, SearchResult

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink
    from rag_core.core_retrieval import SearchRunner
    from rag_core.search.query_plan import QueryPlan
    from rag_core.search.types import Filter


class _RAGCoreRetrievalMethods:
    _event_sink: "EventSink | None"
    _search: "SearchRunner"

    async def search(
        self,
        *,
        query: str,
        namespace: str,
        corpus_ids: list[str],
        limit: int = 10,
        document_ids: list[str] | None = None,
        rerank: bool = True,
        use_lexical_search: bool = True,
        query_plan: "QueryPlan | None" = None,
        metadata_filter: "Filter | None" = None,
        rerank_budget: RerankBudget | None = None,
    ) -> list[SearchResult]:
        from rag_core.core_retrieval import search_with_core

        return await search_with_core(
            search=self._search,
            query=query,
            namespace=namespace,
            corpus_ids=corpus_ids,
            limit=limit,
            document_ids=document_ids,
            rerank=rerank,
            use_lexical_search=use_lexical_search,
            query_plan=query_plan,
            metadata_filter=metadata_filter,
            rerank_budget=rerank_budget,
        )

    async def retrieve_context(
        self,
        *,
        query: str,
        namespace: str,
        corpus_ids: list[str],
        limit: int = 8,
        document_ids: list[str] | None = None,
        rerank: bool = True,
        use_lexical_search: bool = True,
        query_plan: "QueryPlan | None" = None,
        metadata_filter: "Filter | None" = None,
        rerank_budget: RerankBudget | None = None,
        max_chars: int | None = None,
        max_tokens: int | None = None,
    ) -> ModelContextPack:
        from rag_core.core_retrieval import retrieve_context_with_core

        return await retrieve_context_with_core(
            search=self._search,
            event_sink=self._event_sink,
            query=query,
            namespace=namespace,
            corpus_ids=corpus_ids,
            limit=limit,
            document_ids=document_ids,
            rerank=rerank,
            use_lexical_search=use_lexical_search,
            query_plan=query_plan,
            metadata_filter=metadata_filter,
            rerank_budget=rerank_budget,
            max_chars=max_chars,
            max_tokens=max_tokens,
        )
