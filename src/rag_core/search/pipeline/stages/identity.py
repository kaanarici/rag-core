"""No-op pipeline stage adapters."""

from __future__ import annotations

from rag_core.search.pipeline.types import PipelineContext, PipelineQuery
from rag_core.search.vector_models import SearchResult


class IdentityQueryTransform:
    """No-op QueryTransform: returns the query unchanged."""

    async def transform(
        self, query: PipelineQuery, ctx: PipelineContext
    ) -> PipelineQuery:
        return query


class IdentityFuse:
    """Return the first input list unchanged.

    The default Retrieve already produces one fused list (Qdrant's server-side
    RRF), so multi-list fusion is unnecessary in the default pipeline.
    """

    async def fuse(
        self,
        results: list[list[SearchResult]],
        query: PipelineQuery,
        ctx: PipelineContext,
    ) -> list[SearchResult]:
        return results[0] if results else []


class PassThroughRerank:
    """Return input unchanged when no reranker is configured."""

    async def rerank(
        self,
        results: list[SearchResult],
        query: PipelineQuery,
        ctx: PipelineContext,
    ) -> list[SearchResult]:
        return results


class IdentityPostprocess:
    """No-op Postprocess: returns input unchanged."""

    async def postprocess(
        self,
        results: list[SearchResult],
        query: PipelineQuery,
        ctx: PipelineContext,
    ) -> list[SearchResult]:
        return results
