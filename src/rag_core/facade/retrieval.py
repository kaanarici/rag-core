from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from rag_core.retrieval_defaults import (
    DEFAULT_CONTEXT_LIMIT,
    DEFAULT_RERANK,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_USE_LEXICAL_SEARCH,
)
from rag_core.events.types import AuditContext
from rag_core.scope import normalize_namespace, resolve_collections_argument
from rag_core.search import Context, RerankBudget, SearchResult

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink
    from rag_core._engine.core_retrieval import SearchRunner
    from rag_core.search import Filter, QueryPlan


class _EngineRetrievalMethods:
    _event_sink: "EventSink | None"
    _search: "SearchRunner"

    async def search(
        self,
        *,
        query: str,
        collection: str | None = None,
        collections: Sequence[str] | None = None,
        namespace: str | None = None,
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

        resolved_namespace = normalize_namespace(namespace)
        resolved_collections = resolve_collections_argument(
            collection=collection,
            collections=collections,
            caller="Engine.search",
        )
        return await search_with_core(
            search=self._search,
            query=query,
            namespace=resolved_namespace,
            collections=resolved_collections,
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

    async def context(
        self,
        *,
        query: str,
        collection: str | None = None,
        collections: Sequence[str] | None = None,
        namespace: str | None = None,
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
    ) -> Context:
        from rag_core._engine.core_retrieval import context_with_core

        resolved_namespace = normalize_namespace(namespace)
        resolved_collections = resolve_collections_argument(
            collection=collection,
            collections=collections,
            caller="Engine.context",
        )
        return await context_with_core(
            search=self._search,
            event_sink=self._event_sink,
            query=query,
            namespace=resolved_namespace,
            collections=resolved_collections,
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
