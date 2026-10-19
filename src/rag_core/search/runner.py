"""Embed, retrieve, optionally rerank, and trace one scoped search."""

from __future__ import annotations

from rag_core._async import run_sync
import asyncio
from dataclasses import dataclass, replace
from typing import Any, cast
from uuid import uuid4
from rag_core.events.emit import emit_event, now_ms
from rag_core.events.sink import CorrelatedSink, EventSink
from rag_core.events.types import (
    AuditContext,
    EmbedCompleted,
    EmbedRequested,
    SearchCompleted,
    SearchPlanned,
    SearchStageCompleted,
    SearchStarted,
    StageError,
)
from rag_core.events.search_events import RETURNED_DOCUMENT_IDS_CAP
from rag_core.search.context_pack import Context
from rag_core.search.filters import Filter
from rag_core.search.policy import CollectionPolicy
from rag_core.search.provider_protocols import (
    EmbeddingProvider,
    RerankerProvider,
    SparseEmbedder,
    VectorStore,
    provider_name,
)
from rag_core.search.request_models import (
    RerankBudget,
    SearchMode,
    SearchQuery,
    _require_non_blank_string,
    _require_non_blank_string_items,
    _require_positive_int,
    validate_search_mode,
)
from rag_core.search.rerank import rerank_results
from rag_core.search.vector_models import SearchResult, SparseVector


@dataclass(frozen=True)
class SearchRequest:
    query: str
    collections: list[str]
    tenant_id: str
    limit: int = 10
    content_types: list[str] | None = None
    document_ids: list[str] | None = None
    rerank: bool = True
    metadata_filter: Filter | None = None
    rerank_budget: RerankBudget | None = None
    mode: SearchMode = "dense"
    audit_context: AuditContext | None = None

    def __post_init__(self) -> None:
        _require_non_blank_string(self.query, "SearchRequest.query")
        _require_non_blank_string(self.tenant_id, "SearchRequest.tenant_id")
        for name in ("collections", "content_types", "document_ids"):
            _require_non_blank_string_items(
                getattr(self, name), f"SearchRequest.{name}"
            )
        _require_positive_int(self.limit, "SearchRequest.limit")
        validate_search_mode(self.mode)


@dataclass(frozen=True)
class SearchRunResult:
    results: list[SearchResult]
    search_id: str


class SearchRunner:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        sparse_embedder: SparseEmbedder | None,
        vector_store: VectorStore,
        reranker: RerankerProvider | None = None,
        event_sink: EventSink | None = None,
        *,
        collection_policy: CollectionPolicy | None = None,
    ) -> None:
        self._embedding = embedding_provider
        self._sparse = sparse_embedder
        self._store = vector_store
        self._store.bind_embedding_space(embedding_provider.model_name)
        self._reranker = reranker
        self._event_sink = event_sink
        self._collection_policy = collection_policy

    async def search(self, req: SearchRequest) -> list[SearchResult]:
        return (await self.search_with_trace(req)).results

    async def search_with_trace(self, req: SearchRequest) -> SearchRunResult:
        search_id = uuid4().hex
        sink = (
            CorrelatedSink(
                self._event_sink, search_id=search_id, audit_context=req.audit_context
            )
            if self._event_sink
            else None
        )
        if self._collection_policy is not None:
            self._collection_policy.validate_search(
                tenant_id=req.tenant_id, collections=req.collections
            )
        started = now_ms()
        results: list[SearchResult] = []
        attempted = applied = False
        succeeded = False
        emit_event(
            sink,
            SearchStarted(
                tenant_id=req.tenant_id,
                collections=tuple(req.collections),
                query_length=len(req.query),
                limit=req.limit,
            ),
        )
        try:
            if has_empty_allowlist(req):
                succeeded = True
                return SearchRunResult([], search_id)
            await self._store.ensure_collection()
            capabilities = self._store.capabilities
            if req.mode != "dense" and not getattr(capabilities, req.mode):
                raise ValueError(
                    f"{provider_name(self._store)} does not support {req.mode} search"
                )
            native_lexical = capabilities.native_lexical
            needs_sparse = req.mode != "dense" and not native_lexical
            if needs_sparse and self._sparse is None:
                raise ValueError(
                    f"{req.mode} search requires a sparse embedder for this store"
                )
            rerank = req.rerank and self._reranker is not None
            budget = req.rerank_budget or RerankBudget()
            pool = (
                max(req.limit, budget.candidate_count or max(20, req.limit * 4))
                if rerank
                else req.limit
            )
            tasks: list[asyncio.Task[Any]] = []
            if req.mode != "lexical":
                tasks.append(asyncio.create_task(self._embed_dense(req.query, sink)))
            if needs_sparse:
                tasks.append(asyncio.create_task(self._embed_sparse(req.query, sink)))
            try:
                vectors = await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            dense = cast(list[float], vectors[0]) if req.mode != "lexical" else []
            sparse = cast(dict[str, SparseVector], vectors[-1]) if needs_sparse else {}
            query = SearchQuery(
                dense_vector=dense,
                sparse_vectors=sparse,
                tenant_id=req.tenant_id,
                collections=req.collections,
                document_ids=req.document_ids,
                content_types=req.content_types,
                metadata_filter=req.metadata_filter,
                limit=pool,
                mode=req.mode,
                lexical_query=req.query,
                candidate_limit=pool if rerank else None,
            )
            emit_event(
                sink,
                SearchPlanned(
                    tenant_id=req.tenant_id,
                    collections=tuple(req.collections),
                    limit=req.limit,
                    mode=req.mode,
                    candidate_limit=query.candidate_limit or query.limit,
                    rerank_enabled=rerank,
                    channels=("dense:dense:primary", "sparse:bm25:bm25")
                    if req.mode == "hybrid"
                    else (
                        ("sparse:bm25:bm25",)
                        if req.mode == "lexical"
                        else ("dense:dense:primary",)
                    ),
                    fusion="rrf" if req.mode == "hybrid" else "none",
                    metadata_filter=type(req.metadata_filter).__name__
                    if req.metadata_filter is not None
                    else "none",
                    content_type_count=len(req.content_types or []),
                    document_id_count=len(req.document_ids or []),
                    rerank_timeout_ms=(budget.timeout_seconds or 0) * 1000
                    if rerank
                    else 0,
                    rerank_max_output=budget.max_output or 0 if rerank else 0,
                    rerank_fallback_on_error=budget.fallback_on_error
                    if rerank
                    else False,
                ),
            )
            retrieval_started = now_ms()
            results = await self._store.search(query)
            emit_event(
                sink,
                SearchStageCompleted(
                    stage="retrieve",
                    stage_name="search",
                    candidate_count=len(results),
                    result_count=len(results),
                    duration_ms=now_ms() - retrieval_started,
                ),
            )
            if rerank and results:
                attempted = True
                results, applied = await rerank_results(
                    results,
                    query=req.query,
                    limit=req.limit,
                    reranker=self._reranker,
                    budget=budget,
                    event_sink=sink,
                )
            results = results[: req.limit]
            succeeded = True
            return SearchRunResult(results, search_id)
        except BaseException as exc:
            emit_event(sink, StageError(stage="search", error_type=type(exc).__name__))
            raise
        finally:
            emit_event(
                sink,
                SearchCompleted(
                    tenant_id=req.tenant_id,
                    collections=tuple(req.collections),
                    result_count=len(results),
                    requested_rerank=req.rerank,
                    attempted_rerank=attempted,
                    applied_rerank=applied,
                    succeeded=succeeded,
                    duration_ms=now_ms() - started,
                    returned_document_ids=_capped_returned_document_ids(results),
                ),
            )

    async def context(
        self,
        req: SearchRequest,
        *,
        max_chars: int | None = None,
        max_tokens: int | None = None,
        deduplicate: bool = False,
    ) -> Context:
        from rag_core.search.context_pack import build_context_pack

        from rag_core.search.result_collapse import collapse_search_results

        search_request = (
            replace(req, limit=max(20, req.limit * 4)) if deduplicate else req
        )
        run = await self.search_with_trace(search_request)
        results = (
            [
                item.primary
                for item in collapse_search_results(run.results, limit=req.limit)
            ]
            if deduplicate
            else run.results
        )
        sink = (
            CorrelatedSink(
                self._event_sink,
                search_id=run.search_id,
                audit_context=req.audit_context,
            )
            if self._event_sink
            else None
        )
        started = now_ms()
        try:
            pack = build_context_pack(
                results,
                query=req.query,
                max_snippets=req.limit,
                max_chars=max_chars,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            emit_event(
                sink,
                StageError(
                    stage="context_pack",
                    error_type=type(exc).__name__,
                    search_id=run.search_id,
                ),
            )
            raise
        emit_event(
            sink,
            SearchStageCompleted(
                stage="context_pack",
                stage_name="build_context_pack",
                candidate_count=len(run.results),
                result_count=len(pack.snippets),
                dropped_count=pack.dropped_count,
                truncated=pack.truncated,
                max_chars=pack.max_chars or 0,
                max_tokens=pack.max_tokens or 0,
                token_estimate=pack.token_estimate,
                char_count=pack.char_count,
                citation_count=len(pack.citations),
                source_preview_count=len(pack.source_previews),
                duration_ms=now_ms() - started,
                search_id=run.search_id,
            ),
        )
        return pack

    async def _embed_dense(self, text: str, sink: EventSink | None) -> list[float]:
        provider = provider_name(self._embedding)
        model = self._embedding.model_name
        emit_event(
            sink,
            EmbedRequested(provider=provider, model=model, text_count=1, role="dense"),
        )
        started = now_ms()
        vector = await self._embedding.embed_query(text)
        emit_event(
            sink,
            EmbedCompleted(
                provider=provider,
                model=model,
                text_count=1,
                role="dense",
                duration_ms=now_ms() - started,
            ),
        )
        return vector

    async def _embed_sparse(
        self, text: str, sink: EventSink | None
    ) -> dict[str, SparseVector]:
        assert self._sparse is not None
        provider = provider_name(self._sparse)
        model = getattr(self._sparse, "model_name", "")
        emit_event(
            sink,
            EmbedRequested(provider=provider, model=model, text_count=1, role="sparse"),
        )
        started = now_ms()
        vector = await run_sync(self._sparse.embed_query, text)
        emit_event(
            sink,
            EmbedCompleted(
                provider=provider,
                model=model,
                text_count=1,
                role="sparse",
                duration_ms=now_ms() - started,
            ),
        )
        return {"bm25": vector}


def has_empty_allowlist(req: SearchRequest) -> bool:
    return req.collections == [] or req.document_ids == [] or req.content_types == []


def _capped_returned_document_ids(results: list[SearchResult]) -> tuple[str, ...]:
    # Distinct document_ids, in result order, up to RETURNED_DOCUMENT_IDS_CAP.
    # Skips blanks so providers that don't carry document_id don't fill the
    # audit row with empty strings.
    seen: list[str] = []
    seen_set: set[str] = set()
    for hit in results:
        document_id = getattr(hit, "document_id", "") or ""
        if not document_id or document_id in seen_set:
            continue
        seen.append(document_id)
        seen_set.add(document_id)
        if len(seen) >= RETURNED_DOCUMENT_IDS_CAP:
            break
    return tuple(seen)
