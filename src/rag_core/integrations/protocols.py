"""Shared integration protocols for optional retrieval adapters."""

from __future__ import annotations

from typing import Protocol

from rag_core.contracts import SupportsContextPackPromptPayload
from rag_core.search import QueryPlan


class SupportsRetrieveContext(Protocol):
    async def retrieve_context(
        self,
        *,
        query: str,
        namespace: str,
        corpus_ids: list[str],
        limit: int,
        content_types: list[str] | None,
        document_ids: list[str] | None,
        rerank: bool,
        use_lexical_search: bool,
        query_plan: QueryPlan | None,
        max_chars: int | None,
        max_tokens: int | None,
    ) -> SupportsContextPackPromptPayload: ...


__all__ = ["SupportsRetrieveContext"]
