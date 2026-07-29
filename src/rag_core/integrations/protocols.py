"""Shared integration protocols for optional retrieval adapters."""

from __future__ import annotations

from typing import Protocol

from rag_core.contracts import SupportsContextPromptPayload
from rag_core.events.types import AuditContext
from rag_core.search import QueryPlan


class SupportsRetrieveContext(Protocol):
    async def context(
        self,
        *,
        query: str,
        namespace: str,
        collections: list[str],
        limit: int,
        content_types: list[str] | None,
        document_ids: list[str] | None,
        rerank: bool,
        use_lexical_search: bool,
        query_plan: QueryPlan | None,
        max_chars: int | None,
        max_tokens: int | None,
        audit_context: AuditContext | None,
    ) -> SupportsContextPromptPayload: ...


__all__ = [
    "SupportsRetrieveContext",
]
