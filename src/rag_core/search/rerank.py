from __future__ import annotations

from dataclasses import dataclass, replace
from rag_core.events.sink import EventSink
from rag_core.search.provider_protocols import RerankerProvider
from rag_core.search.request_models import RerankBudget

from rag_core.search.result_scores import finite_score_or_zero
from rag_core.search.request_models import RerankResult
from rag_core.search.vector_models import SearchResult


@dataclass(frozen=True)
class AppliedRerankResults:
    ordered: list[SearchResult]
    used_indices: set[int]
    provider_result_count: int
    rank_changed_count: int = 0
    rank_promoted_count: int = 0
    rank_demoted_count: int = 0
    max_rank_gain: int = 0
    max_rank_loss: int = 0
    provider_score_min: float = 0.0
    provider_score_max: float = 0.0
    search_score_min: float = 0.0
    search_score_max: float = 0.0

    @property
    def accepted_count(self) -> int:
        return len(self.used_indices)

    @property
    def dropped_count(self) -> int:
        return self.provider_result_count - self.accepted_count


def apply_rerank_results(
    candidates: list[SearchResult],
    reranked: list[RerankResult],
    *,
    provider: str,
    model: str,
    provider_result_count: int | None = None,
    accepted_limit: int | None = None,
) -> AppliedRerankResults:
    ordered: list[SearchResult] = []
    used_indices: set[int] = set()
    rank_deltas: list[int] = []
    provider_scores: list[float] = []
    search_scores: list[float] = []
    for item in reranked:
        if accepted_limit is not None and len(ordered) >= accepted_limit:
            continue
        if item.index in used_indices or not 0 <= item.index < len(candidates):
            continue
        used_indices.add(item.index)
        original_rank = item.index + 1
        rerank_rank = len(ordered) + 1
        rank_deltas.append(original_rank - rerank_rank)
        provider_scores.append(finite_score_or_zero(item.score))
        search_scores.append(finite_score_or_zero(candidates[item.index].score))
        ordered.append(
            _with_rerank_metadata(
                candidates[item.index],
                provider=provider,
                model=model,
                provider_score=item.score,
                search_score=candidates[item.index].score,
                original_index=item.index,
                rerank_index=len(ordered),
            )
        )
    return AppliedRerankResults(
        ordered=ordered,
        used_indices=used_indices,
        provider_result_count=(
            provider_result_count
            if provider_result_count is not None
            else len(reranked)
        ),
        rank_changed_count=sum(1 for delta in rank_deltas if delta != 0),
        rank_promoted_count=sum(1 for delta in rank_deltas if delta > 0),
        rank_demoted_count=sum(1 for delta in rank_deltas if delta < 0),
        max_rank_gain=max((delta for delta in rank_deltas if delta > 0), default=0),
        max_rank_loss=max((-delta for delta in rank_deltas if delta < 0), default=0),
        provider_score_min=min(provider_scores, default=0.0),
        provider_score_max=max(provider_scores, default=0.0),
        search_score_min=min(search_scores, default=0.0),
        search_score_max=max(search_scores, default=0.0),
    )


def _with_rerank_metadata(
    result: SearchResult,
    *,
    provider: str,
    model: str,
    provider_score: float,
    search_score: float,
    original_index: int,
    rerank_index: int,
) -> SearchResult:
    original_rank = original_index + 1
    rerank_rank = rerank_index + 1
    safe_provider_score = finite_score_or_zero(provider_score)
    safe_search_score = finite_score_or_zero(search_score)
    metadata = dict(result.metadata)
    metadata["rerank"] = {
        "provider": provider,
        "model": model,
        "provider_score": safe_provider_score,
        "search_score": safe_search_score,
        "original_rank": original_rank,
        "rerank_rank": rerank_rank,
        "rank_delta": original_rank - rerank_rank,
    }
    return replace(result, score=safe_search_score, metadata=metadata)


async def rerank_results(
    results: list[SearchResult],
    *,
    query: str,
    limit: int,
    reranker: RerankerProvider | None,
    budget: RerankBudget,
    event_sink: EventSink | None,
) -> tuple[list[SearchResult], bool]:
    assert reranker is not None
    import asyncio
    from rag_core.events.emit import emit_event, now_ms
    from rag_core.events.types import RerankApplied
    from rag_core.search.provider_protocols import provider_name

    candidates = (
        results[: budget.candidate_count] if budget.candidate_count else results
    )
    top_k = min(budget.max_output or limit, len(candidates))
    provider = provider_name(reranker)
    model = getattr(reranker, "model_name", "")
    started = now_ms()
    try:
        call = reranker.rerank(query, [r.text for r in candidates], top_k=top_k)
        ranked = (
            await call
            if budget.timeout_seconds is None
            else await asyncio.wait_for(call, budget.timeout_seconds)
        )
    except (Exception, asyncio.CancelledError) as exc:
        task = asyncio.current_task()
        if (
            isinstance(exc, asyncio.CancelledError)
            and task is not None
            and task.cancelling()
        ):
            raise
        emit_event(
            event_sink,
            RerankApplied(
                provider=provider,
                model=model,
                input_count=len(results),
                candidate_count=len(candidates),
                top_k=top_k,
                result_count=len(results),
                succeeded=False,
                fallback_reason="timeout"
                if isinstance(exc, TimeoutError)
                else type(exc).__name__,
                duration_ms=now_ms() - started,
            ),
        )
        if not budget.fallback_on_error:
            raise
        return results, False
    applied = apply_rerank_results(
        candidates,
        ranked,
        provider=provider,
        model=model,
        provider_result_count=len(ranked),
        accepted_limit=top_k,
    )
    ordered = (
        applied.ordered
        + [r for i, r in enumerate(candidates) if i not in applied.used_indices]
        + results[len(candidates) :]
    )
    emit_event(
        event_sink,
        RerankApplied(
            provider=provider,
            model=model,
            input_count=len(results),
            candidate_count=len(candidates),
            provider_result_count=applied.provider_result_count,
            accepted_count=applied.accepted_count,
            dropped_count=applied.dropped_count,
            result_count=len(ordered),
            top_k=top_k,
            rank_changed_count=applied.rank_changed_count,
            rank_promoted_count=applied.rank_promoted_count,
            rank_demoted_count=applied.rank_demoted_count,
            max_rank_gain=applied.max_rank_gain,
            max_rank_loss=applied.max_rank_loss,
            provider_score_min=applied.provider_score_min,
            provider_score_max=applied.provider_score_max,
            search_score_min=applied.search_score_min,
            search_score_max=applied.search_score_max,
            duration_ms=now_ms() - started,
            succeeded=applied.dropped_count == 0,
        ),
    )
    return ordered, applied.accepted_count > 0
