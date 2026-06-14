from __future__ import annotations

from typing import TYPE_CHECKING

from rag_core.retrieval_defaults import (
    DEFAULT_CONTEXT_LIMIT,
    DEFAULT_RERANK,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_USE_LEXICAL_SEARCH,
)
from rag_core.events.types import AuditContext
from rag_core.search import ContextPack, RerankBudget, SearchResult

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink
    from rag_core._engine.core_retrieval import SearchRunner
    from rag_core.search import Filter, QueryPlan


def _require_explicit_corpus_ids(corpus_ids: object, caller: str) -> None:
    """Refuse a missing or empty corpus_ids list at the facade boundary.

    Fail-closed seam for scoped retrieval: ``None`` or ``[]`` would silently
    widen or erase the requested corpus scope depending on downstream handling.
    Either is a contract bug; raise here instead of returning ``[]`` and
    hiding it.
    """
    if corpus_ids is None:
        raise ValueError(
            f"{caller} requires an explicit corpus_ids list (got None); "
            "silent corpus-scope widening is forbidden"
        )
    if not isinstance(corpus_ids, list) or len(corpus_ids) == 0:
        raise ValueError(
            f"{caller} requires a non-empty corpus_ids list; "
            "pass at least one corpus id"
        )


class _RAGCoreRetrievalMethods:
    _event_sink: "EventSink | None"
    _search: "SearchRunner"

    async def search(
        self,
        *,
        query: str,
        namespace: str,
        corpus_ids: list[str],
        limit: int = DEFAULT_SEARCH_LIMIT,
        content_types: list[str] | None = None,
        document_ids: list[str] | None = None,
        rerank: bool = DEFAULT_RERANK,
        use_lexical_search: bool = DEFAULT_USE_LEXICAL_SEARCH,
        query_plan: "QueryPlan | None" = None,
        metadata_filter: "Filter | None" = None,
        rerank_budget: RerankBudget | None = None,
        audit_context: AuditContext | None = None,
    ) -> list[SearchResult]:
        from rag_core._engine.core_retrieval import search_with_core

        _require_explicit_corpus_ids(corpus_ids, "RAGCore.search")
        return await search_with_core(
            search=self._search,
            query=query,
            namespace=namespace,
            corpus_ids=corpus_ids,
            limit=limit,
            content_types=content_types,
            document_ids=document_ids,
            rerank=rerank,
            use_lexical_search=use_lexical_search,
            query_plan=query_plan,
            metadata_filter=metadata_filter,
            rerank_budget=rerank_budget,
            audit_context=audit_context,
        )

    async def retrieve_context(
        self,
        *,
        query: str,
        namespace: str,
        corpus_ids: list[str],
        limit: int = DEFAULT_CONTEXT_LIMIT,
        content_types: list[str] | None = None,
        document_ids: list[str] | None = None,
        rerank: bool = DEFAULT_RERANK,
        use_lexical_search: bool = DEFAULT_USE_LEXICAL_SEARCH,
        query_plan: "QueryPlan | None" = None,
        metadata_filter: "Filter | None" = None,
        rerank_budget: RerankBudget | None = None,
        max_chars: int | None = None,
        max_tokens: int | None = None,
        audit_context: AuditContext | None = None,
    ) -> ContextPack:
        from rag_core._engine.core_retrieval import retrieve_context_with_core

        _require_explicit_corpus_ids(corpus_ids, "RAGCore.retrieve_context")
        return await retrieve_context_with_core(
            search=self._search,
            event_sink=self._event_sink,
            query=query,
            namespace=namespace,
            corpus_ids=corpus_ids,
            limit=limit,
            content_types=content_types,
            document_ids=document_ids,
            rerank=rerank,
            use_lexical_search=use_lexical_search,
            query_plan=query_plan,
            metadata_filter=metadata_filter,
            rerank_budget=rerank_budget,
            max_chars=max_chars,
            max_tokens=max_tokens,
            audit_context=audit_context,
        )
