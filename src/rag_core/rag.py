from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
import math
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal

from rag_core.contracts import (
    normalize_static_retrieval_scope,
    scope_document_ids,
    validate_bound_namespace,
)
from rag_core.core import Engine
from rag_core.core_models import Config
from rag_core.scope import normalize_collection, normalize_namespace
from rag_core.search.context_pack_sources import (
    source_locator_from_result,
    source_reference_from_result,
)
from rag_core.search.policy import CollectionPolicy
from rag_core.search.query_plan_presets import resolve_prefetch_limit
from rag_core.search.vector_models import SEARCH_RESULT_TYPE_TEXT

TENANT_ID_ENV = "RAG_CORE_TENANT_ID"

if TYPE_CHECKING:
    from rag_core.core_models import DeleteDocumentResult
    from rag_core.events.types import AuditContext
    from rag_core.search import Context, QueryPlan
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


@dataclass(frozen=True)
class _CollapsedSearchResult:
    primary: SearchResult
    equivalents: tuple[SearchResult, ...]


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
        self._tenant_id = validate_bound_namespace(normalize_namespace(tenant_id))
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
            raise ValueError(
                f"RAGCore.from_env requires tenant_id or {TENANT_ID_ENV}"
            )
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
        await self.ensure_ready()
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
            namespace=self._tenant_id,
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
    ) -> RetrievalResult:
        collapsed = await self._search_collapsed(
            query,
            limit=limit,
            document_ids=document_ids,
        )
        return RetrievalResult(
            evidence=tuple(_evidence_from_collapsed(item) for item in collapsed)
        )

    async def _search_collapsed(
        self,
        query: str,
        *,
        limit: int,
        document_ids: Sequence[str] | None,
        audit_context: AuditContext | None = None,
    ) -> tuple[_CollapsedSearchResult, ...]:
        _, requested_document_ids = normalize_static_retrieval_scope(
            collection=self._collection,
            document_ids=document_ids,
            limit=limit,
        )
        scoped_document_ids = scope_document_ids(
            requested=requested_document_ids,
            configured=self._document_ids,
        )
        retrieval_limit = max(
            limit,
            resolve_prefetch_limit(result_limit=limit),
        )
        hits = await self._engine.search(
            query=query,
            namespace=self._tenant_id,
            collection=self._collection,
            limit=retrieval_limit,
            document_ids=scoped_document_ids,
            rerank=False,
            use_lexical_search=False,
            audit_context=audit_context,
        )
        return _collapse_search_results(hits, limit=limit)

    async def delete(self, document_id: str) -> DeleteDocumentResult:
        self._validate_document_id(document_id)
        return await self._engine.delete_document(
            document_id=document_id,
            namespace=self._tenant_id,
            collection=self._collection,
        )

    def tool(self) -> Any:
        """Return an OpenAI Agents SDK tool bound to this facade's scope."""
        from rag_core.integrations.openai_agents import build_retrieve_context_tool

        return build_retrieve_context_tool(
            _RAGCoreContextAdapter(self),
            namespace=self._tenant_id,
            collection=self._collection,
            document_ids=self._document_ids,
            default_rerank=False,
            default_use_lexical_search=False,
            expose_strategy_options=False,
        )

    def _validate_document_id(self, document_id: str) -> None:
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("document_id must be a non-empty string")
        if self._document_ids is not None and document_id not in self._document_ids:
            raise ValueError("document_id is outside the configured document scope")


class _RAGCoreContextAdapter:
    """Prompt-context projection over the facade's canonical retrieval path."""

    def __init__(self, rag: RAGCore) -> None:
        self._rag = rag

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
    ) -> Context:
        if namespace != self._rag.tenant_id or collections != [self._rag.collection]:
            raise ValueError("tool retrieval scope does not match its RAGCore facade")
        if content_types is not None:
            raise ValueError("RAGCore.tool does not expose content-type strategy")
        if rerank or use_lexical_search or query_plan is not None:
            raise ValueError("RAGCore.tool does not expose retrieval strategy")

        from rag_core.search.context_pack import build_context_pack

        collapsed = await self._rag._search_collapsed(
            query,
            limit=limit,
            document_ids=document_ids,
            audit_context=audit_context,
        )
        return build_context_pack(
            (item.primary for item in collapsed),
            query=query,
            max_snippets=limit,
            max_chars=max_chars,
            max_tokens=max_tokens,
        )


def _bind_scope(config: Config, *, tenant_id: str, collection: str) -> Config:
    policy = config.collection_policy or CollectionPolicy()
    policy.validate_namespace(tenant_id)
    policy.validate_collections([collection])
    return replace(
        config,
        collection_policy=replace(
            policy,
            bound_namespace=tenant_id,
        ),
    )


def _validate_engine_scope(engine: Engine, *, tenant_id: str, collection: str) -> None:
    policy = engine._config.collection_policy
    if policy is None:
        return
    policy.validate_namespace(tenant_id)
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


def _collapse_search_results(
    hits: Sequence[SearchResult],
    *,
    limit: int,
) -> tuple[_CollapsedSearchResult, ...]:
    groups: dict[tuple[str, str], list[tuple[int, SearchResult]]] = {}
    for rank, hit in enumerate(hits):
        result_type = hit.result_type or SEARCH_RESULT_TYPE_TEXT
        groups.setdefault((result_type, hit.text), []).append((rank, hit))

    collapsed: list[tuple[int, tuple[str, ...], _CollapsedSearchResult]] = []
    for members in groups.values():
        ordered_hits = sorted(
            (hit for _, hit in members),
            key=_representative_sort_key,
        )
        collapsed.append(
            (
                min(rank for rank, _ in members),
                _stable_source_key(ordered_hits[0]),
                _CollapsedSearchResult(
                    primary=ordered_hits[0],
                    equivalents=tuple(ordered_hits[1:]),
                ),
            )
        )
    collapsed.sort(key=lambda item: (item[0], item[1]))
    return tuple(item[2] for item in collapsed[:limit])


def _representative_sort_key(hit: SearchResult) -> tuple[float, tuple[str, ...]]:
    score = hit.score if math.isfinite(hit.score) else float("-inf")
    return (-score, _stable_source_key(hit))


def _stable_source_key(hit: SearchResult) -> tuple[str, ...]:
    source = source_reference_from_result(hit)
    locator = source_locator_from_result(hit)
    return (
        source.source_id,
        hit.document_id or "",
        hit.document_key or "",
        str(locator.chunk_index if locator.chunk_index is not None else ""),
        str(locator.page_number if locator.page_number is not None else ""),
        str(locator.line_start if locator.line_start is not None else ""),
        str(locator.start_offset if locator.start_offset is not None else ""),
        hit.id,
    )


__all__ = [
    "Document",
    "Evidence",
    "IngestResult",
    "RAGCore",
    "RetrievalResult",
]
