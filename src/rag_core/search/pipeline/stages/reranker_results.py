from __future__ import annotations

from dataclasses import dataclass, replace

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
            provider_result_count if provider_result_count is not None else len(reranked)
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
