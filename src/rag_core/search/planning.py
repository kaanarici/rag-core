from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_core.search.query_plan_presets import (
    DEFAULT_SEARCH_PROFILE,
    QUERY_PLAN_PRESETS,
    QUERY_PLAN_PRESET_DENSE_ONLY,
    QUERY_PLAN_PRESET_SPARSE_ONLY,
    SEARCH_PROFILES,
    default_query_plan,
    default_query_plan_for_capabilities,
    describe_query_plan,
    describe_query_plan_presets,
    describe_search_profile_catalog,
    describe_search_profiles,
    query_plan_preset,
    resolve_prefetch_limit,
    search_profile,
)
from rag_core.search.query_plan import (
    DenseChannel,
    FUSION_KIND_DBSF,
    FUSION_KIND_RRF,
    FUSION_KIND_WEIGHTED_RRF,
    Prefetch,
    PrefetchFusion,
    QueryPlan,
    SparseChannel,
    UnsupportedQueryStage,
)
from rag_core.search.provider_protocols import QueryPlanCapabilities


@runtime_checkable
class QueryPlanValidator(Protocol):
    """Optional adapter hook for provider-specific query-plan shape checks."""

    def validate_query_plan(self, plan: QueryPlan) -> None: ...


@runtime_checkable
class QueryPlanPreparer(Protocol):
    """Optional async hook for provider checks that need store state."""

    async def ensure_collection(self) -> None: ...

    async def prepare_query_plan(self, plan: QueryPlan) -> None: ...


@runtime_checkable
class QueryPlanDefaultProvider(Protocol):
    """Optional adapter hook for store-specific default plans."""

    def default_query_plan(self, *, result_limit: int) -> QueryPlan | None: ...


@runtime_checkable
class QueryPlanSparseVectorPolicy(Protocol):
    """Optional adapter hook for plans whose lexical channel is provider-side."""

    def query_plan_needs_sparse_vectors(self, plan: QueryPlan | None) -> bool: ...


def default_query_plan_for_store(
    *,
    store: object,
    capabilities: QueryPlanCapabilities,
    result_limit: int,
) -> QueryPlan | None:
    if isinstance(store, QueryPlanDefaultProvider):
        return store.default_query_plan(result_limit=result_limit)
    return default_query_plan_for_capabilities(
        capabilities=capabilities,
        result_limit=result_limit,
    )


def validate_query_plan_capabilities(
    plan: QueryPlan,
    *,
    capabilities: QueryPlanCapabilities,
    provider_name: str,
) -> None:
    """Fail before provider work when a declared store cannot run a plan."""
    if not _declares_query_plan_support(capabilities):
        raise UnsupportedQueryStage(
            f"{provider_name} does not declare query-plan support"
        )
    if plan.boost is not None and not capabilities.boost:
        raise UnsupportedQueryStage(
            f"{provider_name} does not support boost query plans"
        )
    if plan.rerank is not None and not capabilities.mmr:
        raise UnsupportedQueryStage(
            f"{provider_name} does not support MMR query plans"
        )
    if plan.fuse is not None:
        _validate_fusion(
            plan.fuse,
            capabilities=capabilities,
            provider_name=provider_name,
        )
    for prefetch in _flatten_prefetches(plan.prefetches):
        if prefetch.nested and not capabilities.nested_prefetch:
            raise UnsupportedQueryStage(
                f"{provider_name} does not support nested query-plan prefetches"
            )
        channel = prefetch.channel
        if isinstance(channel, DenseChannel):
            if not capabilities.dense:
                raise UnsupportedQueryStage(
                    f"{provider_name} does not support dense query-plan channels"
                )
        elif isinstance(channel, SparseChannel) and not capabilities.sparse:
            raise UnsupportedQueryStage(
                f"{provider_name} does not support sparse query-plan channels"
            )


def validate_query_plan_for_store(
    plan: QueryPlan,
    *,
    capabilities: QueryPlanCapabilities,
    provider_name: str,
    store: object,
) -> None:
    """Validate generic capabilities plus an adapter's own query-plan constraints."""
    validate_query_plan_capabilities(
        plan,
        capabilities=capabilities,
        provider_name=provider_name,
    )
    if isinstance(store, QueryPlanValidator):
        store.validate_query_plan(plan)


def _validate_fusion(
    fusion: PrefetchFusion,
    *,
    capabilities: QueryPlanCapabilities,
    provider_name: str,
) -> None:
    if fusion.kind == FUSION_KIND_RRF and not capabilities.hybrid_rrf:
        raise UnsupportedQueryStage(
            f"{provider_name} does not support hybrid RRF query plans"
        )
    if fusion.kind == FUSION_KIND_DBSF and not capabilities.hybrid_dbsf:
        raise UnsupportedQueryStage(
            f"{provider_name} does not support hybrid DBSF query plans"
        )
    if (
        fusion.kind == FUSION_KIND_WEIGHTED_RRF
        and not capabilities.hybrid_weighted_rrf
    ):
        raise UnsupportedQueryStage(
            f"{provider_name} does not support weighted RRF query plans"
        )


def _flatten_prefetches(prefetches: tuple[Prefetch, ...]) -> list[Prefetch]:
    flattened: list[Prefetch] = []
    for prefetch in prefetches:
        flattened.append(prefetch)
        flattened.extend(_flatten_prefetches(prefetch.nested))
    return flattened


def _declares_query_plan_support(capabilities: QueryPlanCapabilities) -> bool:
    return (
        capabilities.dense
        or capabilities.sparse
        or capabilities.hybrid
        or capabilities.mmr
        or capabilities.boost
        or capabilities.nested_prefetch
    )


__all__ = [
    "QUERY_PLAN_PRESETS",
    "QUERY_PLAN_PRESET_DENSE_ONLY",
    "QUERY_PLAN_PRESET_SPARSE_ONLY",
    "SEARCH_PROFILES",
    "QueryPlanPreparer",
    "QueryPlanValidator",
    "QueryPlanDefaultProvider",
    "QueryPlanSparseVectorPolicy",
    "DEFAULT_SEARCH_PROFILE",
    "default_query_plan",
    "default_query_plan_for_capabilities",
    "default_query_plan_for_store",
    "describe_query_plan",
    "describe_query_plan_presets",
    "describe_search_profile_catalog",
    "describe_search_profiles",
    "query_plan_preset",
    "resolve_prefetch_limit",
    "search_profile",
    "validate_query_plan_capabilities",
    "validate_query_plan_for_store",
]
