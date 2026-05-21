"""Named query-plan presets and search profiles."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rag_core.search.query_plan import (
    DenseChannel,
    FusionKind,
    Mmr,
    Prefetch,
    PrefetchFusion,
    QueryPlan,
    SparseChannel,
)
from rag_core.search.sparse_channels import PRIMARY_SPARSE_CHANNEL
from rag_core.search.types import QueryPlanCapabilities

_DEFAULT_PREFETCH_MULTIPLIER = 4
_MIN_PREFETCH_LIMIT = 20
_MAX_PREFETCH_LIMIT = 200
DEFAULT_SEARCH_PROFILE = "balanced"


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
    "hybrid_rrf": QueryPlanPresetSpec(
        summary="dense plus sparse retrieval fused with reciprocal rank fusion",
        channels=("dense", "sparse"),
        fusion="rrf",
    ),
    "dense_only": QueryPlanPresetSpec(
        summary="dense vector retrieval only",
        channels=("dense",),
    ),
    "sparse_only": QueryPlanPresetSpec(
        summary="sparse lexical retrieval only",
        channels=("sparse",),
    ),
    "hybrid_dbsf": QueryPlanPresetSpec(
        summary="dense plus sparse retrieval fused with distribution-based score fusion",
        channels=("dense", "sparse"),
        fusion="dbsf",
    ),
    "hybrid_with_mmr": QueryPlanPresetSpec(
        summary="hybrid reciprocal-rank fusion followed by MMR diversity reranking",
        channels=("dense", "sparse"),
        fusion="rrf",
        rerank="mmr",
    ),
}

SEARCH_PROFILE_SPECS: dict[str, SearchProfileSpec] = {
    "balanced": SearchProfileSpec(
        preset="hybrid_rrf",
        summary="general-purpose hybrid retrieval",
        latency="medium",
        quality="balanced",
    ),
    "fast": SearchProfileSpec(
        preset="dense_only",
        summary="low-latency semantic retrieval",
        latency="low",
        quality="semantic",
    ),
    "lexical": SearchProfileSpec(
        preset="sparse_only",
        summary="keyword-oriented lexical retrieval",
        latency="low",
        quality="lexical",
    ),
    "coverage": SearchProfileSpec(
        preset="hybrid_dbsf",
        summary="hybrid retrieval with score-distribution fusion",
        latency="medium",
        quality="broad",
    ),
    "diverse": SearchProfileSpec(
        preset="hybrid_with_mmr",
        summary="hybrid retrieval with diversity reranking",
        latency="higher",
        quality="diverse",
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
    result_limit: int = 20,
    fusion: FusionKind = "rrf",
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
        rrf_k=60 if fusion in ("rrf", "weighted_rrf") else 0,
    )
    return QueryPlan(
        prefetches=tuple(prefetches),
        fuse=fuse,
        final_limit=result_limit,
    )


def query_plan_preset(name: str, *, limit: int) -> QueryPlan:
    """Build a named ``QueryPlan`` preset for the CLI and other thin surfaces."""
    if name == "hybrid_rrf":
        return default_query_plan(result_limit=limit, fusion="rrf")
    if name == "hybrid_dbsf":
        return default_query_plan(result_limit=limit, fusion="dbsf")
    if name == "dense_only":
        pl = resolve_prefetch_limit(result_limit=limit)
        return QueryPlan(
            prefetches=(Prefetch(channel=DenseChannel(), limit=pl),),
            final_limit=limit,
        )
    if name == "sparse_only":
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
    if name == "hybrid_with_mmr":
        base = default_query_plan(result_limit=limit, fusion="rrf")
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


def describe_retrieval_profiles() -> dict[str, object]:
    return {
        "default_search_profile": DEFAULT_SEARCH_PROFILE,
        "search_profiles": describe_search_profiles(),
        "query_plan_presets": describe_query_plan_presets(),
    }


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


def default_query_plan_for_capabilities(
    *,
    capabilities: QueryPlanCapabilities,
    result_limit: int,
) -> QueryPlan | None:
    """Choose the documented default profile when the store supports it."""
    if capabilities.dense and capabilities.sparse and capabilities.hybrid_rrf:
        return search_profile(DEFAULT_SEARCH_PROFILE, limit=result_limit)
    if capabilities.dense and capabilities.sparse and capabilities.hybrid_dbsf:
        return default_query_plan(result_limit=result_limit, fusion="dbsf")
    if capabilities.dense and capabilities.sparse and capabilities.hybrid_weighted_rrf:
        return default_query_plan(
            result_limit=result_limit,
            fusion="weighted_rrf",
            fusion_weights=(1.0, 1.0),
        )
    if capabilities.dense:
        return query_plan_preset("dense_only", limit=result_limit)
    if capabilities.sparse:
        return query_plan_preset("sparse_only", limit=result_limit)
    return None


__all__ = [
    "DEFAULT_SEARCH_PROFILE",
    "QUERY_PLAN_PRESETS",
    "QUERY_PLAN_PRESET_SPECS",
    "SEARCH_PROFILES",
    "SEARCH_PROFILE_SPECS",
    "default_query_plan",
    "default_query_plan_for_capabilities",
    "describe_query_plan_presets",
    "describe_retrieval_profiles",
    "describe_search_profiles",
    "query_plan_preset",
    "resolve_prefetch_limit",
    "search_profile",
]
