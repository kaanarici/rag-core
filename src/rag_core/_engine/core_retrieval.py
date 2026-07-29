from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import replace
import hashlib
import re
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
from rag_core.search import (
    Answerability,
    Context,
    Filter,
    QueryPlan,
    RerankBudget,
    RetrievalResult,
    SearchResult,
)
from rag_core.search.context_pack import build_context_pack
from rag_core.search.query_plan_presets import (
    QUERY_PLAN_PRESET_DENSE_ONLY,
    QUERY_PLAN_PRESET_HYBRID_RRF,
    QUERY_PLAN_PRESET_SPARSE_ONLY,
    query_plan_preset,
    resolve_prefetch_limit,
)
from rag_core.search.pipeline_runner import (
    SearchExecutionOptions,
    SearchRequest,
    SearchRunResult,
)
from rag_core.search.request_models import SearchOptions


class SearchRunner(Protocol):
    async def search(self, req: SearchRequest) -> list[SearchResult]: ...


async def retrieve_with_core(
    *,
    search: SearchRunner,
    query: str,
    namespace: str,
    collection: str,
    document_ids: list[str] | None,
    limit: int,
    options: SearchOptions,
) -> RetrievalResult:
    candidate_limit = (
        resolve_prefetch_limit(result_limit=limit)
        if (
            options.duplicate_policy == "collapse"
            or options.max_results_per_document is not None
        )
        else limit
    )
    plan = query_plan_preset(
        {
            "dense": QUERY_PLAN_PRESET_DENSE_ONLY,
            "lexical": QUERY_PLAN_PRESET_SPARSE_ONLY,
            "hybrid": QUERY_PLAN_PRESET_HYBRID_RRF,
        }[options.mode],
        limit=candidate_limit,
    )
    candidates = await search_with_core(
        search=search,
        query=query,
        namespace=namespace,
        collections=[collection],
        limit=candidate_limit,
        content_types=(
            list(options.content_types) if options.content_types is not None else None
        ),
        document_ids=document_ids,
        rerank=options.rerank,
        use_lexical_search=False,
        query_plan=plan,
        metadata_filter=options.metadata_filter,
    )
    evidence, duplicate_count, capped_count = _select_evidence(
        candidates,
        limit=limit,
        duplicate_policy=options.duplicate_policy,
        max_results_per_document=options.max_results_per_document,
    )
    top_score = evidence[0].score if evidence else None
    score_margin = (
        evidence[0].score - evidence[1].score if len(evidence) > 1 else None
    )
    answerability_signals: dict[str, object] = {
        "evidence_count": len(evidence),
        "candidate_count": len(candidates),
    }
    if top_score is not None:
        answerability_signals["top_score"] = top_score
    if score_margin is not None:
        answerability_signals["top_score_margin"] = score_margin
    answerability = _answerability(
        top_score=top_score,
        options=options,
        signals=answerability_signals,
    )
    return RetrievalResult(
        evidence=tuple(evidence),
        answerability=answerability,
        diagnostics={
            "mode": options.mode,
            "duplicate_policy": options.duplicate_policy,
            "duplicate_count": duplicate_count,
            "per_document_cap_count": capped_count,
            "candidate_count": len(candidates),
            "limit": limit,
        },
    )


def _answerability(
    *,
    top_score: float | None,
    options: SearchOptions,
    signals: dict[str, object],
) -> Answerability:
    threshold = options.answerability_threshold
    calibration = options.answerability_calibration
    if threshold is None or calibration is None:
        return Answerability(
            status="unknown",
            reason="not_calibrated",
            signals=signals,
        )
    calibrated_signals = {
        **signals,
        "threshold": threshold,
    }
    if top_score is None:
        return Answerability(
            status="insufficient",
            reason="no_evidence",
            calibration=calibration,
            signals=calibrated_signals,
        )
    return Answerability(
        status="sufficient" if top_score >= threshold else "insufficient",
        reason="calibrated_threshold",
        calibration=calibration,
        signals=calibrated_signals,
    )


async def search_with_core(
    *,
    search: SearchRunner,
    query: str,
    namespace: str,
    collections: list[str],
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
            collections=collections,
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


_WHITESPACE_RE = re.compile(r"\s+")


def _select_evidence(
    candidates: list[SearchResult],
    *,
    limit: int,
    duplicate_policy: str,
    max_results_per_document: int | None,
) -> tuple[list[SearchResult], int, int]:
    groups = (
        _exact_duplicate_groups(candidates)
        if duplicate_policy == "collapse"
        else [[candidate] for candidate in candidates]
    )
    selected: list[SearchResult] = []
    per_document: dict[str, int] = {}
    duplicate_count = sum(max(0, len(group) - 1) for group in groups)
    capped_count = 0
    for group in groups:
        retained_index = _first_source_with_capacity(
            group,
            per_document=per_document,
            max_results_per_document=max_results_per_document,
        )
        if retained_index is None:
            capped_count += 1
            continue
        retained = group[retained_index]
        document_key = _evidence_document_key(retained)
        per_document[document_key] = per_document.get(document_key, 0) + 1
        equivalents = tuple(
            _equivalent_source(source)
            for index, source in enumerate(group)
            if index != retained_index
        )
        if equivalents:
            retained = replace(retained, equivalent_sources=equivalents)
        selected.append(retained)
        if len(selected) >= limit:
            break
    return selected, duplicate_count, capped_count


def _exact_duplicate_groups(
    candidates: list[SearchResult],
) -> list[list[SearchResult]]:
    groups: list[list[SearchResult]] = []
    group_indexes: dict[str, int] = {}
    for candidate in candidates:
        identity = hashlib.sha256(
            _WHITESPACE_RE.sub(" ", candidate.text).strip().casefold().encode("utf-8")
        ).hexdigest()
        index = group_indexes.get(identity)
        if index is None:
            group_indexes[identity] = len(groups)
            groups.append([candidate])
        else:
            groups[index].append(candidate)
    return groups


def _first_source_with_capacity(
    group: list[SearchResult],
    *,
    per_document: dict[str, int],
    max_results_per_document: int | None,
) -> int | None:
    if max_results_per_document is None:
        return 0
    for index, source in enumerate(group):
        if (
            per_document.get(_evidence_document_key(source), 0)
            < max_results_per_document
        ):
            return index
    return None


def _evidence_document_key(evidence: SearchResult) -> str:
    return evidence.document_id or evidence.document_key or evidence.id


def _equivalent_source(evidence: SearchResult) -> dict[str, object]:
    values: dict[str, object | None] = {
        "document_id": evidence.document_id,
        "document_key": evidence.document_key,
        "chunk_id": evidence.chunk_id,
        "title": evidence.title,
        "section": evidence.section,
        "locator": evidence.locator,
    }
    return {key: value for key, value in values.items() if value is not None}


async def context_with_core(
    *,
    search: SearchRunner,
    event_sink: EventSink | None,
    query: str,
    namespace: str,
    collections: list[str],
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
) -> Context:
    search_run = await _search_with_trace_if_available(
        search=search,
        query=query,
        namespace=namespace,
        collections=collections,
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
    collections: list[str],
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
        collections=collections,
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
