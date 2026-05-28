"""Pluggable strategies for merging vector results with sidecar results.

`PreferMaxScoreMerge` dedupes by id, keeps the higher score, unions metadata,
and returns the strongest scored results first. Alternative strategies plug in via
`SidecarMergeStrategy`.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Protocol, runtime_checkable

from rag_core.search.result_scores import blended_score, sanitize_result_score
from rag_core.search.stored_payload import merge_duplicate_result
from rag_core.search.vector_models import SearchResult


@runtime_checkable
class SidecarMergeStrategy(Protocol):
    """Combine vector and sidecar results into one ranked list."""

    async def merge(
        self,
        vector_results: list[SearchResult],
        sidecar_results: list[SearchResult],
    ) -> list[SearchResult]: ...


class PreferMaxScoreMerge:
    """Deduplicate sidecar/vector hits and rank by the final sanitized score."""

    async def merge(
        self,
        vector_results: list[SearchResult],
        sidecar_results: list[SearchResult],
    ) -> list[SearchResult]:
        merged: dict[str, tuple[int, SearchResult]] = {}
        next_order = 0
        for result in sidecar_results + vector_results:
            sanitized = sanitize_result_score(result)
            existing = merged.get(sanitized.id)
            if existing is not None:
                order, existing_result = existing
                merged[sanitized.id] = (
                    order,
                    merge_duplicate_result(existing_result, sanitized),
                )
                continue
            merged[sanitized.id] = (next_order, sanitized)
            next_order += 1
        return [
            result
            for _order, result in sorted(
                merged.values(),
                key=lambda item: (-item[1].score, item[0]),
            )
        ]


class PreferSidecarMerge:
    """Sidecar exact matches rank first; fuzzy sidecar rows compete by score.

    Duplicate ids still merge with the sidecar text/metadata winning. Vector-only
    and non-exact sidecar-only rows are ordered by sanitized score after exact
    sidecar hits.
    """

    async def merge(
        self,
        vector_results: list[SearchResult],
        sidecar_results: list[SearchResult],
    ) -> list[SearchResult]:
        vector_by_id = {result.id: result for result in vector_results}
        seen: set[str] = set()
        exact: list[SearchResult] = []
        scored_tail: list[SearchResult] = []
        for sidecar_result in sidecar_results:
            seen.add(sidecar_result.id)
            vector_match = vector_by_id.get(sidecar_result.id)
            merged = (
                sanitize_result_score(sidecar_result)
                if vector_match is None
                else merge_duplicate_result(sidecar_result, vector_match)
            )
            if vector_match is not None or _sidecar_strategy(sidecar_result) == "exact":
                exact.append(merged)
            else:
                scored_tail.append(merged)
        for vector_result in vector_results:
            if vector_result.id in seen:
                continue
            scored_tail.append(sanitize_result_score(vector_result))
        return [
            *exact,
            *sorted(scored_tail, key=_prefer_sidecar_tail_score, reverse=True),
        ]


def _sidecar_strategy(result: SearchResult) -> str | None:
    payload = result.metadata.get("search_sidecar")
    if isinstance(payload, dict):
        value = payload.get("strategy")
        if isinstance(value, str):
            return value
    return None


def _prefer_sidecar_tail_score(result: SearchResult) -> float:
    rerank = result.metadata.get("rerank")
    if isinstance(rerank, dict):
        provider_score = rerank.get("provider_score")
        if (
            not isinstance(provider_score, bool)
            and isinstance(provider_score, int | float)
            and math.isfinite(float(provider_score))
        ):
            return float(provider_score)
    return result.score


class ScoreBlendMerge:
    """Linear blend on duplicates: alpha * vector_score + (1 - alpha) * sidecar_score.

    alpha=1 keeps the vector score; alpha=0 keeps the sidecar score; 0.5 averages.
    Non-duplicate results pass through with their original score. Sidecar-only
    rows lead, then unique vector rows.
    """

    def __init__(self, alpha: float = 0.5) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self._alpha = alpha

    async def merge(
        self,
        vector_results: list[SearchResult],
        sidecar_results: list[SearchResult],
    ) -> list[SearchResult]:
        vector_by_id = {result.id: result for result in vector_results}
        ordered: list[SearchResult] = []
        seen: set[str] = set()
        for sidecar_result in sidecar_results:
            seen.add(sidecar_result.id)
            vector_match = vector_by_id.get(sidecar_result.id)
            if vector_match is None:
                ordered.append(sanitize_result_score(sidecar_result))
                continue
            blended = blended_score(
                vector_score=vector_match.score,
                sidecar_score=sidecar_result.score,
                alpha=self._alpha,
            )
            ordered.append(replace(sidecar_result, score=blended))
        for vector_result in vector_results:
            if vector_result.id in seen:
                continue
            ordered.append(sanitize_result_score(vector_result))
        return ordered
