"""Linear retrieval pipeline.

The pipeline is the canonical seam for advanced and experimental retrieval
techniques (HyDE, multi-query, MMR, reranker cascades, parent-child expansion,
etc.). See `docs/adr/0002-linear-pipeline-no-dsl.md`. Stages are typed
protocols; the runner is a frozen dataclass with five fields.
"""

from __future__ import annotations

from rag_core.search.pipeline.merge_strategies import (
    PreferMaxScoreMerge,
    PreferSidecarMerge,
    ScoreBlendMerge,
    SidecarMergeStrategy,
)
from rag_core.search.pipeline.runner import RetrievalPipeline
from rag_core.search.pipeline.stages.hybrid_retrieve import HybridRetrieve
from rag_core.search.pipeline.stages.identity import (
    IdentityFuse,
    IdentityPostprocess,
    IdentityQueryTransform,
    PassThroughRerank,
)
from rag_core.search.pipeline.stages.reranker_stage import ProviderRerankStage
from rag_core.search.pipeline.stages.sidecar_postprocess import (
    SidecarPostprocess,
    SidecarPrefetchTransform,
)
from rag_core.search.pipeline.types import (
    FuseStage,
    PipelineContext,
    PipelineQuery,
    Postprocess,
    QueryTransform,
    Rerank,
    Retrieve,
)

__all__ = (
    "FuseStage",
    "HybridRetrieve",
    "IdentityFuse",
    "IdentityPostprocess",
    "IdentityQueryTransform",
    "PassThroughRerank",
    "PipelineContext",
    "PipelineQuery",
    "Postprocess",
    "PreferMaxScoreMerge",
    "PreferSidecarMerge",
    "ProviderRerankStage",
    "QueryTransform",
    "Rerank",
    "Retrieve",
    "RetrievalPipeline",
    "ScoreBlendMerge",
    "SidecarMergeStrategy",
    "SidecarPostprocess",
    "SidecarPrefetchTransform",
)
