"""Client-side reciprocal-rank fusion stage."""

from __future__ import annotations

from dataclasses import dataclass, replace

from rag_core.search.pipeline.types import PipelineContext, PipelineQuery
from rag_core.search.vector_models import SearchResult


@dataclass(frozen=True)
class RrfFuse:
    k: int = 60

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError("RrfFuse.k must be positive")

    async def fuse(
        self,
        results: list[list[SearchResult]],
        query: PipelineQuery,
        ctx: PipelineContext,
    ) -> list[SearchResult]:
        del query, ctx
        scores: dict[str, float] = {}
        first_seen: dict[str, tuple[int, int, str]] = {}
        kept: dict[str, SearchResult] = {}
        for list_index, result_list in enumerate(results):
            seen_in_list: set[str] = set()
            for rank, result in enumerate(result_list, start=1):
                identity = _result_identity(result)
                if identity in seen_in_list:
                    continue
                seen_in_list.add(identity)
                scores[identity] = scores.get(identity, 0.0) + 1.0 / (self.k + rank)
                if identity not in kept:
                    kept[identity] = result
                    first_seen[identity] = (list_index, rank, identity)
        ordered = sorted(
            kept,
            key=lambda identity: (-scores[identity], first_seen[identity]),
        )
        return [replace(kept[identity], score=scores[identity]) for identity in ordered]


def _result_identity(result: SearchResult) -> str:
    return result.id


__all__ = ["RrfFuse"]
