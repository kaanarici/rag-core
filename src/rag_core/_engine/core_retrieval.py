from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol, cast

from rag_core.events.emit import emit_event, now_ms
from rag_core.events.sink import EventSink
from rag_core.events.trace_payload_fields import CONTEXT_PACK_SEARCH_STAGE
from rag_core.events.trace_summary_models import safe_search_id
from rag_core.events.types import AuditContext, SearchStageCompleted, StageError
from rag_core.retrieval_defaults import (
    DEFAULT_CONTEXT_LIMIT,
    DEFAULT_RERANK,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_USE_LEXICAL_SEARCH,
)
from rag_core.search import ContextPack, Filter, QueryPlan, RerankBudget, SearchResult
from rag_core.search.context_pack import build_context_pack
from rag_core.search.pipeline_runner import (
    SearchExecutionOptions,
    SearchRequest,
    SearchRunResult,
)


class SearchRunner(Protocol):
    async def search(self, req: SearchRequest) -> list[SearchResult]: ...


async def search_with_core(
    *,
    search: SearchRunner,
    query: str,
    namespace: str,
    corpus_ids: list[str],
    limit: int = DEFAULT_SEARCH_LIMIT,
    content_types: list[str] | None = None,
    document_ids: list[str] | None = None,
    rerank: bool = DEFAULT_RERANK,
    use_lexical_search: bool = DEFAULT_USE_LEXICAL_SEARCH,
    query_plan: QueryPlan | None = None,
    metadata_filter: Filter | None = None,
    rerank_budget: RerankBudget | None = None,
    audit_context: AuditContext | None = None,
) -> list[SearchResult]:
    return await search.search(
        SearchRequest(
            query=query,
            corpus_ids=corpus_ids,
            namespace=namespace,
            limit=limit,
            content_types=content_types,
            document_ids=document_ids,
            rerank=rerank,
            metadata_filter=metadata_filter,
            rerank_budget=rerank_budget,
            execution=SearchExecutionOptions(
                use_lexical_search=use_lexical_search,
                query_plan=query_plan,
            ),
            audit_context=audit_context,
        )
    )


async def retrieve_context_with_core(
    *,
    search: SearchRunner,
    event_sink: EventSink | None,
    query: str,
    namespace: str,
    corpus_ids: list[str],
    limit: int = DEFAULT_CONTEXT_LIMIT,
    content_types: list[str] | None = None,
    document_ids: list[str] | None = None,
    rerank: bool = DEFAULT_RERANK,
    use_lexical_search: bool = DEFAULT_USE_LEXICAL_SEARCH,
    query_plan: QueryPlan | None = None,
    metadata_filter: Filter | None = None,
    rerank_budget: RerankBudget | None = None,
    max_chars: int | None = None,
    max_tokens: int | None = None,
    audit_context: AuditContext | None = None,
) -> ContextPack:
    search_run = await _search_with_trace_if_available(
        search=search,
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
    hits = search_run.results
    started_ms = now_ms()
    safe_search_identifier = safe_search_id(search_run.search_id)
    try:
        pack = build_context_pack(
            hits,
            query=query,
            max_snippets=_context_pack_limit(limit=limit, query_plan=query_plan),
            max_chars=max_chars,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        emit_event(
            event_sink,
            StageError(
                stage=CONTEXT_PACK_SEARCH_STAGE,
                error_type=type(exc).__name__,
                search_id=safe_search_identifier,
            ),
        )
        raise
    emit_event(
        event_sink,
        SearchStageCompleted(
            stage=CONTEXT_PACK_SEARCH_STAGE,
            stage_name="build_context_pack",
            candidate_count=len(hits),
            result_count=len(pack.snippets),
            dropped_count=pack.dropped_count,
            truncated=pack.truncated,
            max_chars=pack.max_chars or 0,
            max_tokens=pack.max_tokens or 0,
            token_estimate=pack.token_estimate,
            char_count=pack.char_count,
            citation_count=len(pack.citations),
            source_preview_count=len(pack.source_previews),
            duration_ms=now_ms() - started_ms,
            search_id=safe_search_identifier,
        ),
    )
    return pack


def _context_pack_limit(*, limit: int, query_plan: QueryPlan | None) -> int:
    if query_plan is None:
        return limit
    return max(limit, query_plan.final_limit)


async def _search_with_trace_if_available(
    *,
    search: SearchRunner,
    query: str,
    namespace: str,
    corpus_ids: list[str],
    limit: int,
    content_types: list[str] | None,
    document_ids: list[str] | None,
    rerank: bool,
    use_lexical_search: bool,
    query_plan: QueryPlan | None,
    metadata_filter: Filter | None,
    rerank_budget: RerankBudget | None,
    audit_context: AuditContext | None = None,
) -> SearchRunResult:
    request = SearchRequest(
        query=query,
        corpus_ids=corpus_ids,
        namespace=namespace,
        limit=limit,
        content_types=content_types,
        document_ids=document_ids,
        rerank=rerank,
        metadata_filter=metadata_filter,
        rerank_budget=rerank_budget,
        execution=SearchExecutionOptions(
            use_lexical_search=use_lexical_search,
            query_plan=query_plan,
        ),
        audit_context=audit_context,
    )
    search_with_trace = getattr(search, "search_with_trace", None)
    if callable(search_with_trace):
        traced_search = cast(
            Callable[[SearchRequest], Awaitable[SearchRunResult]],
            search_with_trace,
        )
        return await traced_search(request)
    return SearchRunResult(results=await search.search(request), search_id="")
