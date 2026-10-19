"""Shared integration protocols for optional retrieval adapters."""

from __future__ import annotations
from rag_core.search.request_models import SearchMode

from typing import Protocol
from rag_core.search.context_pack import Context

from rag_core.events.types import AuditContext


class SupportsRetrieveContext(Protocol):
    async def context(
        self,
        *,
        query: str,
        tenant_id: str,
        collections: list[str],
        limit: int,
        content_types: list[str] | None,
        document_ids: list[str] | None,
        rerank: bool,
        mode: SearchMode | None = None,
        max_chars: int | None,
        max_tokens: int | None,
        audit_context: AuditContext | None,
    ) -> Context: ...


__all__ = [
    "SupportsRetrieveContext",
]
