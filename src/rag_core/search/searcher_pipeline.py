"""Default pipeline wiring for the search orchestrator."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag_core.search.pipeline import (
    HybridRetrieve,
    IdentityFuse,
    PassThroughRerank,
    PipelineQuery,
    Postprocess,
    ProviderRerankStage,
    QueryTransform,
    RetrievalPipeline,
    SidecarPostprocess,
    SidecarPrefetchTransform,
)
if TYPE_CHECKING:
    from rag_core.search.searcher import SearchRequest


def use_sidecar_for_request(req: SearchRequest) -> bool:
    """Use sidecar only for implicit default planning, not explicit query plans."""
    if not req.use_sidecar:
        return False
    return req.query_plan is None


def default_search_pipeline(
    *,
    reranker_present: bool,
    sidecar_present: bool,
) -> RetrievalPipeline:
    transforms: tuple[QueryTransform, ...] = ()
    postprocesses: tuple[Postprocess, ...] = ()
    if sidecar_present:
        transforms = (SidecarPrefetchTransform(),)
        postprocesses = (SidecarPostprocess(),)
    return RetrievalPipeline(
        retrieve=HybridRetrieve(),
        fuse=IdentityFuse(),
        rerank=ProviderRerankStage() if reranker_present else PassThroughRerank(),
        query_transforms=transforms,
        postprocesses=postprocesses,
    )


def pipeline_query_from_search_request(
    req: SearchRequest,
    *,
    use_sidecar: bool | None = None,
) -> PipelineQuery:
    return PipelineQuery(
        query=req.query,
        namespace=req.namespace,
        corpus_ids=req.corpus_ids,
        limit=_effective_request_limit(req),
        document_ids=req.document_ids,
        content_types=req.content_types,
        rerank=req.rerank,
        use_sidecar=use_sidecar_for_request(req) if use_sidecar is None else use_sidecar,
        query_plan=req.query_plan,
        query_vector=req.query_vector,
        query_sparse_vectors=req.query_sparse_vectors,
        metadata_filter=req.metadata_filter,
        rerank_budget=req.rerank_budget,
    )


def _effective_request_limit(req: SearchRequest) -> int:
    if req.query_plan is None:
        return req.limit
    return req.query_plan.final_limit
