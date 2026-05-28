"""Named query-plan presets and search profiles."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from rag_core.retrieval_channels import (
    DENSE_RETRIEVAL_CHANNEL,
    SPARSE_RETRIEVAL_CHANNEL,
)
from rag_core.retrieval_defaults import DEFAULT_SEARCH_LIMIT
from rag_core.search.query_plan import (
    DEFAULT_RRF_K,
    DenseChannel,
    FUSION_KIND_DBSF,
    FUSION_KIND_RRF,
    FUSION_KIND_WEIGHTED_RRF,
    FusionKind,
    Mmr,
    Prefetch,
    PrefetchFusion,
    QueryPlan,
    SparseChannel,
)
from rag_core.search.provider_protocols import QueryPlanCapabilities
from rag_core.search.sparse_channels import PRIMARY_SPARSE_CHANNEL

_DEFAULT_PREFETCH_MULTIPLIER = 4
_MIN_PREFETCH_LIMIT = 20
_MAX_PREFETCH_LIMIT = 200
QUERY_PLAN_PRESET_HYBRID_RRF: Final[str] = "hybrid_rrf"
QUERY_PLAN_PRESET_DENSE_ONLY: Final[str] = "dense_only"
QUERY_PLAN_PRESET_SPARSE_ONLY: Final[str] = "sparse_only"
QUERY_PLAN_PRESET_HYBRID_DBSF: Final[str] = "hybrid_dbsf"
QUERY_PLAN_PRESET_HYBRID_WITH_MMR: Final[str] = "hybrid_with_mmr"
QUERY_PLAN_RERANK_MMR: Final[str] = "mmr"

SEARCH_PROFILE_BALANCED: Final[str] = "balanced"
SEARCH_PROFILE_FAST: Final[str] = "fast"
SEARCH_PROFILE_LEXICAL: Final[str] = "lexical"
SEARCH_PROFILE_COVERAGE: Final[str] = "coverage"
SEARCH_PROFILE_DIVERSE: Final[str] = "diverse"

DEFAULT_SEARCH_PROFILE = SEARCH_PROFILE_BALANCED
DEFAULT_SEARCH_PROFILE_SCOPE = "catalog_profile_when_supported"
DEFAULT_QUERY_PLAN_BEHAVIOR = "capability_aware"
DEFAULT_SEARCH_PROFILE_NOTE = (
    f"{SEARCH_PROFILE_BALANCED} is the catalog default when the active vector store supports "
    "dense+sparse hybrid RRF; capability-aware defaults may fall back to "
    f"{QUERY_PLAN_PRESET_DENSE_ONLY} or {QUERY_PLAN_PRESET_SPARSE_ONLY}"
)


@dataclass(frozen=True)
class QueryPlanPresetSpec:
    summary: str
    channels: tuple[str, ...]
    fusion: FusionKind | None = None
    rerank: str | None = None


@dataclass(frozen=True)
class SearchProfileSpec:
    preset: str
    summary: str
    latency: str
    quality: str


QUERY_PLAN_PRESET_SPECS: dict[str, QueryPlanPresetSpec] = {
    QUERY_PLAN_PRESET_HYBRID_RRF: QueryPlanPresetSpec(
        summary="dense plus sparse retrieval fused with reciprocal rank fusion",
        channels=(DENSE_RETRIEVAL_CHANNEL, SPARSE_RETRIEVAL_CHANNEL),
        fusion=FUSION_KIND_RRF,
    ),
    QUERY_PLAN_PRESET_DENSE_ONLY: QueryPlanPresetSpec(
        summary="dense vector retrieval only",
        channels=(DENSE_RETRIEVAL_CHANNEL,),
    ),
    QUERY_PLAN_PRESET_SPARSE_ONLY: QueryPlanPresetSpec(
        summary="sparse lexical retrieval only",
        channels=(SPARSE_RETRIEVAL_CHANNEL,),
    ),
    QUERY_PLAN_PRESET_HYBRID_DBSF: QueryPlanPresetSpec(
        summary="dense plus sparse retrieval fused with distribution-based score fusion",
        channels=(DENSE_RETRIEVAL_CHANNEL, SPARSE_RETRIEVAL_CHANNEL),
        fusion=FUSION_KIND_DBSF,
    ),
    QUERY_PLAN_PRESET_HYBRID_WITH_MMR: QueryPlanPresetSpec(
        summary="hybrid reciprocal-rank fusion followed by MMR diversity reranking",
        channels=(DENSE_RETRIEVAL_CHANNEL, SPARSE_RETRIEVAL_CHANNEL),
        fusion=FUSION_KIND_RRF,
        rerank=QUERY_PLAN_RERANK_MMR,
    ),
}

SEARCH_PROFILE_SPECS: dict[str, SearchProfileSpec] = {
    SEARCH_PROFILE_BALANCED: SearchProfileSpec(
        preset=QUERY_PLAN_PRESET_HYBRID_RRF,
        summary="general-purpose hybrid retrieval",
        latency="medium",
        quality=SEARCH_PROFILE_BALANCED,
    ),
    SEARCH_PROFILE_FAST: SearchProfileSpec(
        preset=QUERY_PLAN_PRESET_DENSE_ONLY,
        summary="low-latency semantic retrieval",
        latency="low",
        quality="semantic",
    ),
    SEARCH_PROFILE_LEXICAL: SearchProfileSpec(
        preset=QUERY_PLAN_PRESET_SPARSE_ONLY,
        summary="keyword-oriented lexical retrieval",
        latency="low",
        quality=SEARCH_PROFILE_LEXICAL,
    ),
    SEARCH_PROFILE_COVERAGE: SearchProfileSpec(
        preset=QUERY_PLAN_PRESET_HYBRID_DBSF,
        summary="hybrid retrieval with score-distribution fusion",
        latency="medium",
        quality="broad",
    ),
    SEARCH_PROFILE_DIVERSE: SearchProfileSpec(
        preset=QUERY_PLAN_PRESET_HYBRID_WITH_MMR,
        summary="hybrid retrieval with diversity reranking",
        latency="higher",
        quality=SEARCH_PROFILE_DIVERSE,
    ),
}

QUERY_PLAN_PRESETS: tuple[str, ...] = tuple(QUERY_PLAN_PRESET_SPECS)
SEARCH_PROFILES: tuple[str, ...] = tuple(SEARCH_PROFILE_SPECS)


def resolve_prefetch_limit(*, result_limit: int, requested: int | None = None) -> int:
    if result_limit <= 0:
        raise ValueError("limit must be positive")
    if requested is not None:
        if requested < result_limit:
            raise ValueError("prefetch_limit must be greater than or equal to limit")
        return requested
    return min(
        max(result_limit * _DEFAULT_PREFETCH_MULTIPLIER, _MIN_PREFETCH_LIMIT),
        _MAX_PREFETCH_LIMIT,
    )


def default_query_plan(
    *,
    result_limit: int = DEFAULT_SEARCH_LIMIT,
    fusion: FusionKind = FUSION_KIND_RRF,
    prefetch_limit: int | None = None,
    fusion_weights: Sequence[float] = (),
    sparse_channels: Sequence[str] = (PRIMARY_SPARSE_CHANNEL,),
) -> QueryPlan:
    """Build a basic dense+sparse hybrid plan."""
    pl = resolve_prefetch_limit(result_limit=result_limit, requested=prefetch_limit)
    prefetches: list[Prefetch] = [Prefetch(channel=DenseChannel(), limit=pl)]
    for channel_name in sparse_channels:
        prefetches.append(
            Prefetch(
                channel=SparseChannel(
                    vector_field=channel_name,
                    using_query_vector=channel_name,
                ),
                limit=pl,
            )
        )
    fuse = PrefetchFusion(
        kind=fusion,
        weights=tuple(fusion_weights),
        rrf_k=(
            DEFAULT_RRF_K
            if fusion in (FUSION_KIND_RRF, FUSION_KIND_WEIGHTED_RRF)
            else 0
        ),
    )
    return QueryPlan(
        prefetches=tuple(prefetches),
        fuse=fuse,
        final_limit=result_limit,
    )


def query_plan_preset(name: str, *, limit: int) -> QueryPlan:
    """Build a named ``QueryPlan`` preset for the CLI and other thin surfaces."""
    if name == QUERY_PLAN_PRESET_HYBRID_RRF:
        return default_query_plan(result_limit=limit, fusion=FUSION_KIND_RRF)
    if name == QUERY_PLAN_PRESET_HYBRID_DBSF:
        return default_query_plan(result_limit=limit, fusion=FUSION_KIND_DBSF)
    if name == QUERY_PLAN_PRESET_DENSE_ONLY:
        pl = resolve_prefetch_limit(result_limit=limit)
        return QueryPlan(
            prefetches=(Prefetch(channel=DenseChannel(), limit=pl),),
            final_limit=limit,
        )
    if name == QUERY_PLAN_PRESET_SPARSE_ONLY:
        pl = resolve_prefetch_limit(result_limit=limit)
        return QueryPlan(
            prefetches=(
                Prefetch(
                    channel=SparseChannel(
                        vector_field=PRIMARY_SPARSE_CHANNEL,
                        using_query_vector=PRIMARY_SPARSE_CHANNEL,
                    ),
                    limit=pl,
                ),
            ),
            final_limit=limit,
        )
    if name == QUERY_PLAN_PRESET_HYBRID_WITH_MMR:
        base = default_query_plan(result_limit=limit, fusion=FUSION_KIND_RRF)
        candidate_limit = max(prefetch.limit for prefetch in base.prefetches)
        return QueryPlan(
            prefetches=base.prefetches,
            fuse=base.fuse,
            rerank=Mmr(diversity=0.5, limit=candidate_limit),
            final_limit=limit,
        )
    valid = ", ".join(QUERY_PLAN_PRESETS)
    raise ValueError(f"unknown query plan preset {name!r}; valid presets: {valid}")


def search_profile(name: str, *, limit: int) -> QueryPlan:
    """Build a named search profile over the canonical ``QueryPlan`` presets."""
    spec = SEARCH_PROFILE_SPECS.get(name)
    if spec is None:
        valid = ", ".join(SEARCH_PROFILES)
        raise ValueError(f"unknown search profile {name!r}; valid profiles: {valid}")
    plan = query_plan_preset(spec.preset, limit=limit)
    return QueryPlan(
        prefetches=plan.prefetches,
        fuse=plan.fuse,
        rerank=plan.rerank,
        boost=plan.boost,
        final_limit=plan.final_limit,
        search_profile=name,
    )


def describe_search_profile_catalog(
    *,
    capabilities: QueryPlanCapabilities | None = None,
    result_limit: int = DEFAULT_SEARCH_LIMIT,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "default_search_profile": DEFAULT_SEARCH_PROFILE,
        "default_search_profile_scope": DEFAULT_SEARCH_PROFILE_SCOPE,
        "default_query_plan_behavior": DEFAULT_QUERY_PLAN_BEHAVIOR,
        "default_search_profile_note": DEFAULT_SEARCH_PROFILE_NOTE,
        "search_profiles": describe_search_profiles(),
        "query_plan_presets": describe_query_plan_presets(),
    }
    if capabilities is not None:
        payload["effective_default_query_plan"] = describe_query_plan(
            default_query_plan_for_capabilities(
                capabilities=capabilities,
                result_limit=result_limit,
            )
        )
    return payload


def describe_search_profiles() -> dict[str, dict[str, object]]:
    return {
        name: {
            "preset": spec.preset,
            "summary": spec.summary,
            "latency": spec.latency,
            "quality": spec.quality,
            "default": name == DEFAULT_SEARCH_PROFILE,
        }
        for name, spec in SEARCH_PROFILE_SPECS.items()
    }


def describe_query_plan_presets() -> dict[str, dict[str, object]]:
    return {
        name: {
            "summary": spec.summary,
            "channels": list(spec.channels),
            "fusion": spec.fusion,
            "rerank": spec.rerank,
        }
        for name, spec in QUERY_PLAN_PRESET_SPECS.items()
    }


def describe_query_plan(plan: QueryPlan | None) -> dict[str, object] | None:
    if plan is None:
        return None
    channels = [
        DENSE_RETRIEVAL_CHANNEL
        if isinstance(prefetch.channel, DenseChannel)
        else SPARSE_RETRIEVAL_CHANNEL
        for prefetch in plan.prefetches
    ]
    return {
        "search_profile": plan.search_profile,
        "channels": channels,
        "fusion": plan.fuse.kind if plan.fuse is not None else None,
        "rerank": QUERY_PLAN_RERANK_MMR if plan.rerank is not None else None,
        "final_limit": plan.final_limit,
    }


def default_query_plan_for_capabilities(
    *,
    capabilities: QueryPlanCapabilities,
    result_limit: int,
) -> QueryPlan | None:
    """Choose the documented default profile when the store supports it."""
    if capabilities.dense and capabilities.sparse and capabilities.hybrid_rrf:
        return search_profile(DEFAULT_SEARCH_PROFILE, limit=result_limit)
    if capabilities.dense and capabilities.sparse and capabilities.hybrid_dbsf:
        return default_query_plan(result_limit=result_limit, fusion=FUSION_KIND_DBSF)
    if capabilities.dense and capabilities.sparse and capabilities.hybrid_weighted_rrf:
        return default_query_plan(
            result_limit=result_limit,
            fusion=FUSION_KIND_WEIGHTED_RRF,
            fusion_weights=(1.0, 1.0),
        )
    if capabilities.dense:
        return query_plan_preset(QUERY_PLAN_PRESET_DENSE_ONLY, limit=result_limit)
    if capabilities.sparse:
        return query_plan_preset(QUERY_PLAN_PRESET_SPARSE_ONLY, limit=result_limit)
    return None


__all__ = [
    "DEFAULT_QUERY_PLAN_BEHAVIOR",
    "DEFAULT_SEARCH_PROFILE",
    "DEFAULT_SEARCH_PROFILE_NOTE",
    "DEFAULT_SEARCH_PROFILE_SCOPE",
    "QUERY_PLAN_PRESETS",
    "QUERY_PLAN_PRESET_DENSE_ONLY",
    "QUERY_PLAN_PRESET_HYBRID_DBSF",
    "QUERY_PLAN_PRESET_HYBRID_RRF",
    "QUERY_PLAN_PRESET_HYBRID_WITH_MMR",
    "QUERY_PLAN_PRESET_SPARSE_ONLY",
    "QUERY_PLAN_PRESET_SPECS",
    "QUERY_PLAN_RERANK_MMR",
    "SEARCH_PROFILES",
    "SEARCH_PROFILE_BALANCED",
    "SEARCH_PROFILE_COVERAGE",
    "SEARCH_PROFILE_DIVERSE",
    "SEARCH_PROFILE_FAST",
    "SEARCH_PROFILE_LEXICAL",
    "SEARCH_PROFILE_SPECS",
    "default_query_plan",
    "default_query_plan_for_capabilities",
    "describe_query_plan_presets",
    "describe_query_plan",
    "describe_search_profile_catalog",
    "describe_search_profiles",
    "query_plan_preset",
    "resolve_prefetch_limit",
    "search_profile",
]
