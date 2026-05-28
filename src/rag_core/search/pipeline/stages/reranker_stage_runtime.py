"""Runtime planning helpers for provider-backed rerank stages."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from rag_core.search.providers.rerank_results import (
    rerank_provider_result_count as rerank_provider_result_count,
)
from rag_core.search.provider_protocols import RerankerProvider
from rag_core.search.request_models import RerankBudget, RerankResult
from rag_core.search.vector_models import SearchResult


@dataclass(frozen=True)
class ProviderRerankRun:
    provider: str
    model: str
    budget: RerankBudget
    candidates: list[SearchResult]
    tail: list[SearchResult]
    top_k: int
    truncation_reason: str


def build_provider_rerank_run(
    *,
    results: list[SearchResult],
    reranker: RerankerProvider,
    limit: int,
    budget: RerankBudget,
) -> ProviderRerankRun:
    candidate_count = _candidate_count(results, budget)
    candidates = results[:candidate_count]
    top_k = _top_k(limit, candidate_count, budget)
    return ProviderRerankRun(
        provider=_provider_name(reranker),
        model=getattr(reranker, "model_name", ""),
        budget=budget,
        candidates=candidates,
        tail=results[candidate_count:],
        top_k=top_k,
        truncation_reason=_truncation_reason(
            input_count=len(results),
            candidate_count=candidate_count,
            top_k=top_k,
            budget=budget,
        ),
    )


async def run_reranker(
    *,
    reranker: RerankerProvider,
    query: str,
    candidates: list[SearchResult],
    top_k: int,
    timeout_seconds: float | None,
) -> list[RerankResult]:
    call = reranker.rerank(
        query,
        [result.text for result in candidates],
        top_k=top_k,
    )
    if timeout_seconds is None:
        return await call
    return await asyncio.wait_for(call, timeout=timeout_seconds)


def rerank_fallback_reason(exc: BaseException) -> str:
    if isinstance(exc, TimeoutError):
        return "timeout"
    return type(exc).__name__


def _provider_name(provider: object) -> str:
    name = getattr(provider, "provider_name", None)
    if isinstance(name, str) and name:
        return name
    return type(provider).__name__


def _candidate_count(results: list[SearchResult], budget: RerankBudget) -> int:
    if budget.candidate_count is None:
        return len(results)
    return min(budget.candidate_count, len(results))


def _top_k(limit: int, candidate_count: int, budget: RerankBudget) -> int:
    requested = budget.max_output or limit
    return min(requested, candidate_count)


def _truncation_reason(
    *,
    input_count: int,
    candidate_count: int,
    top_k: int,
    budget: RerankBudget,
) -> str:
    reasons: list[str] = []
    if candidate_count < input_count:
        reasons.append("candidate_count")
    if top_k < candidate_count:
        reasons.append("max_output" if budget.max_output is not None else "query_limit")
    return ",".join(reasons) if reasons else "none"
