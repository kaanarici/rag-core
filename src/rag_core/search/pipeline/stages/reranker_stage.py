"""Wraps a `RerankerProvider` as a Rerank stage."""

from __future__ import annotations

import asyncio
import logging

from rag_core.events.emit import emit_event, now_ms
from rag_core.events.types import RerankApplied
from rag_core.search.pipeline.types import PipelineContext, PipelineQuery
from rag_core.search.pipeline.stages.reranker_results import (
    apply_rerank_results,
)
from rag_core.search.pipeline.stages.reranker_stage_runtime import (
    build_provider_rerank_run,
    rerank_provider_result_count,
    rerank_fallback_reason,
    run_reranker,
)
from rag_core.search.request_models import RerankBudget
from rag_core.search.vector_models import SearchResult

logger = logging.getLogger(__name__)


class ProviderRerankStage:
    """Re-rank candidates using `ctx.reranker`. No-op if absent or empty."""

    real_rerank = True

    async def rerank(
        self,
        results: list[SearchResult],
        query: PipelineQuery,
        ctx: PipelineContext,
    ) -> list[SearchResult]:
        if ctx.reranker is None or not results:
            return results
        sink = ctx.event_sink
        reranker = ctx.reranker
        run = build_provider_rerank_run(
            results=results,
            reranker=reranker,
            limit=query.limit,
            budget=query.rerank_budget or RerankBudget(),
        )
        ctx.execution.attempted_rerank = True
        started_ms = now_ms()
        try:
            reranked = await run_reranker(
                reranker=reranker,
                query=query.query,
                candidates=run.candidates,
                top_k=run.top_k,
                timeout_seconds=run.budget.timeout_seconds,
            )
        except asyncio.CancelledError as exc:
            if _is_current_task_cancelling():
                raise
            fallback_reason = rerank_fallback_reason(exc)
            if run.budget.fallback_on_error:
                logger.warning(
                    "Reranking failed for %s with %s; returning search results without reranking",
                    run.provider,
                    type(exc).__name__,
                )
            emit_event(
                sink,
                RerankApplied(
                    provider=run.provider,
                    model=run.model,
                    input_count=len(results),
                    candidate_count=len(run.candidates),
                    provider_result_count=0,
                    accepted_count=0,
                    dropped_count=0,
                    result_count=len(results),
                    top_k=run.top_k,
                    fallback_reason=fallback_reason,
                    truncation_reason=run.truncation_reason,
                    duration_ms=now_ms() - started_ms,
                    succeeded=False,
                ),
            )
            if not run.budget.fallback_on_error:
                raise
            return results
        except Exception as exc:
            fallback_reason = rerank_fallback_reason(exc)
            if run.budget.fallback_on_error:
                logger.warning(
                    "Reranking failed for %s with %s; returning search results without reranking",
                    run.provider,
                    type(exc).__name__,
                )
            emit_event(
                sink,
                RerankApplied(
                    provider=run.provider,
                    model=run.model,
                    input_count=len(results),
                    candidate_count=len(run.candidates),
                    provider_result_count=0,
                    accepted_count=0,
                    dropped_count=0,
                    result_count=len(results),
                    top_k=run.top_k,
                    fallback_reason=fallback_reason,
                    truncation_reason=run.truncation_reason,
                    duration_ms=now_ms() - started_ms,
                    succeeded=False,
                ),
            )
            if not run.budget.fallback_on_error:
                raise
            return results
        applied = apply_rerank_results(
            run.candidates,
            reranked,
            provider=run.provider,
            model=run.model,
            provider_result_count=rerank_provider_result_count(reranked),
            accepted_limit=run.top_k,
        )
        ordered = applied.ordered
        used_indices = applied.used_indices
        if not ordered:
            ordered = run.candidates
            used_indices = set(range(len(run.candidates)))
        remaining_candidates = [
            result
            for index, result in enumerate(run.candidates)
            if index not in used_indices
        ]
        final_results = ordered + remaining_candidates + run.tail
        emit_event(
            sink,
            RerankApplied(
                provider=run.provider,
                model=run.model,
                input_count=len(results),
                candidate_count=len(run.candidates),
                provider_result_count=applied.provider_result_count,
                accepted_count=applied.accepted_count,
                dropped_count=applied.dropped_count,
                rank_changed_count=applied.rank_changed_count,
                rank_promoted_count=applied.rank_promoted_count,
                rank_demoted_count=applied.rank_demoted_count,
                max_rank_gain=applied.max_rank_gain,
                max_rank_loss=applied.max_rank_loss,
                provider_score_min=applied.provider_score_min,
                provider_score_max=applied.provider_score_max,
                search_score_min=applied.search_score_min,
                search_score_max=applied.search_score_max,
                result_count=len(final_results),
                top_k=run.top_k,
                truncation_reason=run.truncation_reason,
                duration_ms=now_ms() - started_ms,
                succeeded=applied.dropped_count == 0,
            ),
        )
        ctx.execution.applied_rerank = applied.accepted_count > 0
        return final_results


def _is_current_task_cancelling() -> bool:
    task = asyncio.current_task()
    if task is None:
        return False
    return task.cancelling() > 0
