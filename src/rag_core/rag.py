from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal

from rag_core.contracts import (
    normalize_static_retrieval_scope,
    scope_document_ids,
    validate_bound_tenant_id,
)
from rag_core.core import Engine
from rag_core.core_models import Config
from rag_core.scope import normalize_collection, normalize_tenant_id
from rag_core.search.context_pack_sources import (
    source_locator_from_result,
    source_reference_from_result,
)
from rag_core.search.policy import CollectionPolicy
from rag_core.search.result_collapse import (
    CollapsedSearchResult as _CollapsedSearchResult,
    collapse_search_results as _collapse_search_results,
)

TENANT_ID_ENV = "RAG_CORE_TENANT_ID"

if TYPE_CHECKING:
    from rag_core.core_models import DeleteDocumentResult
    from rag_core.events.types import AuditContext
    from rag_core.search import Context
    from rag_core.search.context_pack_models import SourceLocator
    from rag_core.search.vector_models import SearchResult


@dataclass(frozen=True, kw_only=True)
class Document:
    """One application-owned source artifact."""

    key: str
    content: bytes
    content_type: str
    id: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("Document.key must be a non-empty string")
        object.__setattr__(self, "key", self.key.strip())
        if not isinstance(self.content, bytes):
            raise ValueError("Document.content must be bytes")
        if not isinstance(self.content_type, str) or not self.content_type.strip():
            raise ValueError("Document.content_type must be a non-empty string")
        object.__setattr__(self, "content_type", self.content_type.strip())
        if self.id is not None:
            if not isinstance(self.id, str) or not self.id.strip():
                raise ValueError("Document.id must be a non-empty string when set")
            object.__setattr__(self, "id", self.id.strip())
        if not isinstance(self.metadata, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.metadata.items()
        ):
            raise ValueError("Document.metadata must map strings to strings")


IngestStatus = Literal["created", "unchanged", "replaced"]


@dataclass(frozen=True, kw_only=True)
class IngestResult:
    """Outcome of ingesting one application-owned document."""

    document_id: str
    status: IngestStatus
    chunk_count: int
    content_hash: str | None


@dataclass(frozen=True, kw_only=True)
class Evidence:
    """One ranked chunk with stable source identity and location."""

    chunk_id: str
    source_id: str
    text: str
    score: float
    locator: SourceLocator
    document_id: str | None = None
    document_key: str | None = None
    title: str | None = None
    section: str | None = None
    content_type: str | None = None
    source_type: str | None = None
    equivalent_sources: tuple[Evidence, ...] = ()


@dataclass(frozen=True, kw_only=True)
class RetrievalResult:
    """Ranked evidence returned by the scoped facade."""

    evidence: tuple[Evidence, ...]


class RAGCore:
    """Scoped application facade over the retrieval engine."""

    def __init__(
        self,
        config_or_engine: Config | Engine,
        *,
        tenant_id: str,
        collection: str,
        document_ids: Sequence[str] | None = None,
    ) -> None:
        self._tenant_id = validate_bound_tenant_id(normalize_tenant_id(tenant_id))
        self._collection = normalize_collection(collection)
        _, self._document_ids = normalize_static_retrieval_scope(
            collection=self._collection,
            document_ids=document_ids,
            limit=1,
        )
        if isinstance(config_or_engine, Config):
            self._engine = Engine(
                _bind_scope(
                    config_or_engine,
                    tenant_id=self._tenant_id,
                    collection=self._collection,
                )
            )
        elif isinstance(config_or_engine, Engine):
            _validate_engine_scope(
                config_or_engine,
                tenant_id=self._tenant_id,
                collection=self._collection,
            )
            self._engine = config_or_engine
        else:
            raise TypeError("RAGCore requires a Config or Engine")

    @classmethod
    def from_env(
        cls,
        *,
        collection: str,
        tenant_id: str | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> RAGCore:
        """Build the scoped facade from strict process environment settings."""
        from rag_core.config.env_access import get_env_optional
        from rag_core.config.env_config import build_config_from_env

        resolved_tenant = tenant_id
        if resolved_tenant is None:
            resolved_tenant = get_env_optional(TENANT_ID_ENV)
        if resolved_tenant is None or not resolved_tenant.strip():
            raise ValueError(f"RAGCore.from_env requires tenant_id or {TENANT_ID_ENV}")
        return cls(
            build_config_from_env(),
            tenant_id=resolved_tenant,
            collection=collection,
            document_ids=document_ids,
        )

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    @property
    def collection(self) -> str:
        return self._collection

    async def __aenter__(self) -> RAGCore:
        await self._engine.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def ensure_ready(self) -> None:
        await self._engine.ensure_ready()

    async def close(self) -> None:
        await self._engine.close()

    async def ingest(self, document: Document) -> IngestResult:
        if self._document_ids is not None and document.id is None:
            raise ValueError(
                "Document.id is required when RAGCore has a configured document scope"
            )
        if document.id is not None:
            self._validate_document_id(document.id)
        ingested = await self._engine.add_bytes(
            file_bytes=document.content,
            filename=Path(document.key).name or document.key,
            mime_type=document.content_type,
            tenant_id=self._tenant_id,
            collection=self._collection,
            document_id=document.id,
            document_key=document.key,
            path=document.key,
            metadata=document.metadata,
        )
        status: IngestStatus
        if ingested.ingest_state in {"replaced", "reindexed"}:
            status = "replaced"
        elif ingested.ingest_state == "created":
            status = "created"
        elif ingested.ingest_state == "unchanged":
            status = "unchanged"
        else:
            raise RuntimeError(f"unexpected ingest state: {ingested.ingest_state}")
        return IngestResult(
            document_id=ingested.document_id,
            status=status,
            chunk_count=ingested.chunk_count,
            content_hash=ingested.content_sha256,
        )

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        document_ids: Sequence[str] | None = None,
        rerank: bool = False,
    ) -> RetrievalResult:
        collapsed = await self._search_collapsed(
            query,
            limit=limit,
            document_ids=document_ids,
            rerank=rerank,
        )
        return RetrievalResult(
            evidence=tuple(_evidence_from_collapsed(item) for item in collapsed)
        )

    async def context(
        self,
        query: str,
        *,
        limit: int = 5,
        document_ids: Sequence[str] | None = None,
        max_chars: int | None = None,
        max_tokens: int | None = None,
        rerank: bool = False,
        audit_context: AuditContext | None = None,
    ) -> Context:
        """Return prompt-ready context through the same search and scope rules."""
        return await self._engine.context(
            query=query,
            tenant_id=self._tenant_id,
            collection=self._collection,
            limit=limit,
            document_ids=self._scoped_document_ids(document_ids, limit),
            rerank=rerank,
            deduplicate=True,
            max_chars=max_chars,
            max_tokens=max_tokens,
            audit_context=audit_context,
        )

    def _scoped_document_ids(
        self, document_ids: Sequence[str] | None, limit: int
    ) -> list[str] | None:
        _, requested = normalize_static_retrieval_scope(
            collection=self._collection, document_ids=document_ids, limit=limit
        )
        return scope_document_ids(requested=requested, configured=self._document_ids)

    async def _search_collapsed(
        self,
        query: str,
        *,
        limit: int,
        document_ids: Sequence[str] | None,
        audit_context: AuditContext | None = None,
        rerank: bool = False,
    ) -> tuple[_CollapsedSearchResult, ...]:
        scoped_document_ids = self._scoped_document_ids(document_ids, limit)
        retrieval_limit = max(
            limit,
            max(20, limit * 4),
        )
        hits = await self._engine.search(
            query=query,
            tenant_id=self._tenant_id,
            collection=self._collection,
            limit=retrieval_limit,
            document_ids=scoped_document_ids,
            rerank=rerank,
            audit_context=audit_context,
        )
        return _collapse_search_results(hits, limit=limit)

    async def delete(self, document_id: str) -> DeleteDocumentResult:
        self._validate_document_id(document_id)
        return await self._engine.delete_document(
            document_id=document_id,
            tenant_id=self._tenant_id,
            collection=self._collection,
        )

    def tool(self) -> Any:
        """Return a scope-bound async Python function for any agent framework."""

        async def search_user_documents(
            query: str, limit: int = 5
        ) -> dict[str, object]:
            """Search app-owned documents and return context with citations."""
            pack = await self.context(query, limit=limit)
            return {
                **pack.to_prompt_payload(),
                "ok": True,
                "context_text": pack.as_prompt_text(),
            }

        return search_user_documents

    def _validate_document_id(self, document_id: str) -> None:
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("document_id must be a non-empty string")
        if self._document_ids is not None and document_id not in self._document_ids:
            raise ValueError("document_id is outside the configured document scope")


def _bind_scope(config: Config, *, tenant_id: str, collection: str) -> Config:
    policy = config.collection_policy or CollectionPolicy()
    policy.validate_tenant_id(tenant_id)
    policy.validate_collections([collection])
    return replace(
        config,
        collection_policy=replace(
            policy,
            bound_tenant_id=tenant_id,
        ),
    )


def _validate_engine_scope(engine: Engine, *, tenant_id: str, collection: str) -> None:
    policy = engine._config.collection_policy
    if policy is None:
        return
    policy.validate_tenant_id(tenant_id)
    policy.validate_collections([collection])


def _evidence_from_hit(hit: SearchResult) -> Evidence:
    source = source_reference_from_result(hit)
    return Evidence(
        chunk_id=hit.id,
        source_id=source.source_id,
        document_id=hit.document_id,
        document_key=hit.document_key,
        text=hit.text,
        score=hit.score,
        title=hit.title,
        section=hit.section_title or hit.section_path or hit.section_id,
        content_type=hit.content_type,
        source_type=hit.source_type,
        locator=source_locator_from_result(hit),
    )


def _evidence_from_collapsed(item: _CollapsedSearchResult) -> Evidence:
    representative = _evidence_from_hit(item.primary)
    equivalents = tuple(_evidence_from_hit(hit) for hit in item.equivalents)
    return replace(representative, equivalent_sources=equivalents)


__all__ = [
    "Document",
    "Evidence",
    "IngestResult",
    "RAGCore",
    "RetrievalResult",
]
