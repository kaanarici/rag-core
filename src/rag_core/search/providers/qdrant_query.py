"""Query path for the Qdrant vector store.

Channel translation, the typed :class:`QueryPlan` -> Qdrant Query API
translator, store-side request guards, and search execution.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as rest

from rag_core.search.planning import (
    DEFAULT_SEARCH_PROFILE,
    QUERY_PLAN_PRESET_DENSE_ONLY,
    default_query_plan,
    query_plan_preset,
    search_profile,
    validate_query_plan_capabilities,
)
from rag_core.search.policy import VectorStorePolicy
from rag_core.search.query_plan import (
    BOOST_KIND_EXP_DECAY,
    BOOST_KIND_GAUSS_DECAY,
    BOOST_KIND_LINEAR_DECAY,
    BOOST_KIND_RAW,
    Boost,
    DEFAULT_RRF_K,
    DenseChannel,
    FUSION_KIND_DBSF,
    FUSION_KIND_RRF,
    FUSION_KIND_WEIGHTED_RRF,
    Mmr,
    Prefetch,
    PrefetchFusion,
    QueryPlan,
    SparseChannel,
    UnsupportedQueryStage,
)
from rag_core.search.request_models import DeleteFilter, SearchQuery
from rag_core.search.vector_models import SearchResult

from .qdrant_filters import build_search_filter
from .qdrant_payloads import (
    _DENSE_VECTOR_NAME,
    _KNOWN_SPARSE_VECTOR_NAMES,
    _PRIMARY_SPARSE_VECTOR_NAME,
    _point_to_result,
)
from .vector_dimensions import validate_query_dense_dimensions
from .vector_store_capabilities import (
    QDRANT_VECTOR_STORE_CAPABILITY_SPEC,
    QDRANT_VECTOR_STORE_PROVIDER_SPEC,
)


def query_for_qdrant_channel(channel: object, query: SearchQuery) -> Any:
    if isinstance(channel, DenseChannel):
        return query.dense_vector
    if isinstance(channel, SparseChannel):
        sparse = query.all_sparse_vectors().get(channel.using_query_vector)
        if sparse is None:
            raise UnsupportedQueryStage(
                f"SparseChannel({channel.using_query_vector!r}) has no matching sparse query vector"
            )
        return rest.SparseVector(indices=sparse.indices, values=sparse.values)
    raise UnsupportedQueryStage(f"Unknown channel type: {type(channel).__name__}")


def using_for_qdrant_channel(channel: object) -> str:
    if isinstance(channel, DenseChannel):
        return _DENSE_VECTOR_NAME if channel.vector_field == "" else channel.vector_field
    if isinstance(channel, SparseChannel):
        return channel.vector_field
    raise UnsupportedQueryStage(f"Unknown channel type: {type(channel).__name__}")


def ensure_qdrant_sparse_channel_supported(
    channel: object,
    available_sparse_names: frozenset[str] | set[str],
) -> None:
    if isinstance(channel, SparseChannel):
        if channel.vector_field not in available_sparse_names:
            available = ", ".join(sorted(available_sparse_names)) or "none"
            raise UnsupportedQueryStage(
                f"SparseChannel({channel.vector_field!r}) is not available in this "
                f"Qdrant collection; available sparse channels: {available}"
            )
        return
    if isinstance(channel, DenseChannel):
        return
    raise UnsupportedQueryStage(f"Unknown channel type: {type(channel).__name__}")


@dataclass(frozen=True)
class TranslatedPlan:
    """Result of translating a ``QueryPlan`` into Qdrant Query API arguments."""

    prefetch: list[rest.Prefetch]
    query: Any
    limit: int
    using: str | None = None
    query_filter: rest.Filter | None = None


_DECAY_PARAM_KEYS = frozenset({"x", "target", "scale", "midpoint", "defaults"})
_RAW_PARAM_KEYS = frozenset({"formula", "defaults"})


def validate_qdrant_query_plan_shape(plan: QueryPlan) -> None:
    """Reject Qdrant-specific plan shapes the generic capability flags cannot express."""
    if (
        len(plan.prefetches) == 1
        and plan.prefetches[0].nested
        and plan.fuse is None
        and plan.rerank is None
    ):
        raise UnsupportedQueryStage(
            "Qdrant nested prefetch plans require a Fuse or MMR rerank stage"
        )
    validate_qdrant_query_plan_known_sparse_channels(plan)


def validate_qdrant_query_plan_known_sparse_channels(plan: QueryPlan) -> None:
    for prefetch in plan.prefetches:
        _validate_prefetch_known_sparse_channel(prefetch)


def _validate_prefetch_known_sparse_channel(prefetch: Prefetch) -> None:
    channel = prefetch.channel
    if (
        isinstance(channel, SparseChannel)
        and channel.vector_field not in _KNOWN_SPARSE_VECTOR_NAMES
    ):
        supported = ", ".join(sorted(_KNOWN_SPARSE_VECTOR_NAMES))
        raise UnsupportedQueryStage(
            f"Qdrant supports sparse channels {supported}; got {channel.vector_field!r}"
        )
    for nested in prefetch.nested:
        _validate_prefetch_known_sparse_channel(nested)


def validate_qdrant_query_plan_sparse_channels(
    plan: QueryPlan,
    *,
    available_sparse_names: frozenset[str] | set[str],
) -> None:
    for prefetch in plan.prefetches:
        _validate_prefetch_sparse_channels(
            prefetch,
            available_sparse_names=available_sparse_names,
        )


def _validate_prefetch_sparse_channels(
    prefetch: Prefetch,
    *,
    available_sparse_names: frozenset[str] | set[str],
) -> None:
    ensure_qdrant_sparse_channel_supported(prefetch.channel, available_sparse_names)
    for nested in prefetch.nested:
        _validate_prefetch_sparse_channels(
            nested,
            available_sparse_names=available_sparse_names,
        )


def _boost_defaults(
    raw_defaults: object,
    *,
    kind: str,
) -> dict[str, object] | None:
    if raw_defaults is None:
        return None
    if not isinstance(raw_defaults, Mapping):
        raise UnsupportedQueryStage(
            f"Boost(kind={kind!r}) defaults must be a mapping of string keys"
        )
    defaults: dict[str, object] = {}
    for key, value in raw_defaults.items():
        if not isinstance(key, str):
            raise UnsupportedQueryStage(
                f"Boost(kind={kind!r}) defaults keys must be strings"
            )
        defaults[key] = value
    return defaults


def _optional_positive_float(value: object, *, name: str, kind: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UnsupportedQueryStage(
            f"Boost(kind={kind!r}) param {name!r} must be a number"
        )
    numeric = float(value)
    if name == "scale" and numeric <= 0.0:
        raise UnsupportedQueryStage(f"Boost(kind={kind!r}) param 'scale' must be > 0")
    if name == "midpoint" and not 0.0 < numeric < 1.0:
        raise UnsupportedQueryStage(
            f"Boost(kind={kind!r}) param 'midpoint' must be in (0, 1)"
        )
    return numeric


def _qdrant_formula_query(boost: Boost) -> rest.FormulaQuery:
    if boost.kind == BOOST_KIND_RAW:
        unknown = sorted(set(boost.params) - _RAW_PARAM_KEYS)
        if unknown:
            raise UnsupportedQueryStage(
                "Boost(kind='raw') supports only params {'formula', 'defaults'}; "
                f"got unexpected keys: {unknown}"
            )
        raw_formula_obj = boost.params.get("formula")
        if raw_formula_obj is None:
            raise UnsupportedQueryStage("Boost(kind='raw') requires params['formula']")
        raw_formula = cast(rest.Expression, raw_formula_obj)
        return rest.FormulaQuery(
            formula=raw_formula,
            defaults=_boost_defaults(boost.params.get("defaults"), kind=boost.kind),
        )

    unknown = sorted(set(boost.params) - _DECAY_PARAM_KEYS)
    if unknown:
        raise UnsupportedQueryStage(
            f"Boost(kind={boost.kind!r}) supports only params "
            "{'x', 'target', 'scale', 'midpoint', 'defaults'}; "
            f"got unexpected keys: {unknown}"
        )

    x_expr = cast(rest.Expression, boost.params.get("x", boost.field))
    target_expr = cast(rest.Expression | None, boost.params.get("target"))
    params = rest.DecayParamsExpression(
        x=x_expr,
        target=target_expr,
        scale=_optional_positive_float(
            boost.params.get("scale"),
            name="scale",
            kind=boost.kind,
        ),
        midpoint=_optional_positive_float(
            boost.params.get("midpoint"),
            name="midpoint",
            kind=boost.kind,
        ),
    )

    if boost.kind == BOOST_KIND_LINEAR_DECAY:
        decay_expression: rest.Expression = rest.LinDecayExpression(lin_decay=params)
    elif boost.kind == BOOST_KIND_EXP_DECAY:
        decay_expression = rest.ExpDecayExpression(exp_decay=params)
    elif boost.kind == BOOST_KIND_GAUSS_DECAY:
        decay_expression = rest.GaussDecayExpression(gauss_decay=params)
    else:
        raise UnsupportedQueryStage(f"Unknown boost kind: {boost.kind!r}")

    return rest.FormulaQuery(
        formula=rest.SumExpression(sum=["$score", decay_expression]),
        defaults=_boost_defaults(boost.params.get("defaults"), kind=boost.kind),
    )


def _translate_prefetch(
    prefetch: Prefetch,
    *,
    query: SearchQuery,
    qdrant_filter: rest.Filter,
    available_sparse_names: frozenset[str] | set[str],
) -> rest.Prefetch:
    ensure_qdrant_sparse_channel_supported(prefetch.channel, available_sparse_names)
    nested = [
        _translate_prefetch(
            nested_prefetch,
            query=query,
            qdrant_filter=qdrant_filter,
            available_sparse_names=available_sparse_names,
        )
        for nested_prefetch in prefetch.nested
    ]

    return rest.Prefetch(
        query=query_for_qdrant_channel(prefetch.channel, query),
        using=using_for_qdrant_channel(prefetch.channel),
        limit=prefetch.limit,
        filter=qdrant_filter,
        prefetch=nested or None,
    )


def _translate_fuse(fuse: PrefetchFusion, num_prefetches: int) -> Any:
    if fuse.kind == FUSION_KIND_RRF:
        if fuse.rrf_k == DEFAULT_RRF_K:
            return rest.FusionQuery(fusion=rest.Fusion.RRF)
        return rest.RrfQuery(rrf=rest.Rrf(k=fuse.rrf_k))
    if fuse.kind == FUSION_KIND_DBSF:
        return rest.FusionQuery(fusion=rest.Fusion.DBSF)
    if fuse.kind == FUSION_KIND_WEIGHTED_RRF:
        if len(fuse.weights) != num_prefetches:
            raise UnsupportedQueryStage(
                "PrefetchFusion(weighted_rrf) requires one weight per prefetch "
                f"(got {len(fuse.weights)} weights for {num_prefetches} prefetches)"
            )
        return rest.RrfQuery(rrf=rest.Rrf(k=fuse.rrf_k, weights=list(fuse.weights)))
    raise UnsupportedQueryStage(f"Unknown fusion kind: {fuse.kind!r}")


def _translate_rerank(
    rerank: Mmr,
    *,
    query: SearchQuery,
) -> Any:
    if isinstance(rerank, Mmr):
        return rest.NearestQuery(
            nearest=query.dense_vector,
            mmr=rest.Mmr(diversity=rerank.diversity, candidates_limit=rerank.limit),
        )
    raise UnsupportedQueryStage(f"Unknown rerank type: {type(rerank).__name__}")


def translate_query_plan(
    plan: QueryPlan,
    *,
    query: SearchQuery,
    qdrant_filter: rest.Filter,
    available_sparse_names: frozenset[str] | set[str] = _KNOWN_SPARSE_VECTOR_NAMES,
) -> TranslatedPlan:
    """Convert a :class:`QueryPlan` into Qdrant ``query_points`` arguments.

    Stages that the installed Qdrant client cannot honor raise
    :class:`UnsupportedQueryStage` with a message identifying the stage.
    """
    base_plan = QueryPlan(
        prefetches=plan.prefetches,
        fuse=plan.fuse,
        rerank=plan.rerank,
        final_limit=plan.final_limit,
    )
    translated_base = _translate_plan_without_boost(
        base_plan,
        query=query,
        qdrant_filter=qdrant_filter,
        available_sparse_names=available_sparse_names,
    )
    if plan.boost is None:
        return translated_base

    wrapped_prefetch = rest.Prefetch(
        prefetch=translated_base.prefetch or None,
        query=translated_base.query,
        using=translated_base.using,
        limit=translated_base.limit,
        filter=qdrant_filter,
    )
    return TranslatedPlan(
        prefetch=[wrapped_prefetch],
        query=_qdrant_formula_query(plan.boost),
        limit=plan.final_limit,
    )


def _translate_plan_without_boost(
    plan: QueryPlan,
    *,
    query: SearchQuery,
    qdrant_filter: rest.Filter,
    available_sparse_names: frozenset[str] | set[str] = _KNOWN_SPARSE_VECTOR_NAMES,
) -> TranslatedPlan:
    validate_qdrant_query_plan_shape(plan)

    inner_prefetches = [
        _translate_prefetch(
            prefetch,
            query=query,
            qdrant_filter=qdrant_filter,
            available_sparse_names=available_sparse_names,
        )
        for prefetch in plan.prefetches
    ]

    if plan.rerank is not None and plan.fuse is None and len(plan.prefetches) > 1:
        raise UnsupportedQueryStage(
            "Plan with multiple prefetches and a rerank stage requires a Fuse"
        )

    if plan.fuse is not None and len(inner_prefetches) > 1:
        fuse_query = _translate_fuse(plan.fuse, num_prefetches=len(inner_prefetches))
        if plan.rerank is None:
            return TranslatedPlan(
                prefetch=inner_prefetches,
                query=fuse_query,
                limit=plan.final_limit,
            )

        rerank_query = _translate_rerank(plan.rerank, query=query)
        wrapped_prefetch = rest.Prefetch(
            prefetch=inner_prefetches,
            query=fuse_query,
            limit=max(plan.rerank.limit, plan.final_limit),
            filter=qdrant_filter,
        )
        return TranslatedPlan(
            prefetch=[wrapped_prefetch],
            query=rerank_query,
            limit=plan.final_limit,
        )

    if len(inner_prefetches) == 1 and plan.rerank is None:
        single_prefetch = plan.prefetches[0]
        return TranslatedPlan(
            prefetch=[],
            query=rest.NearestQuery(
                nearest=query_for_qdrant_channel(single_prefetch.channel, query),
            ),
            limit=plan.final_limit,
            using=using_for_qdrant_channel(single_prefetch.channel),
            query_filter=qdrant_filter,
        )

    if len(inner_prefetches) == 1 and plan.rerank is not None:
        rerank_query = _translate_rerank(plan.rerank, query=query)
        return TranslatedPlan(
            prefetch=inner_prefetches,
            query=rerank_query,
            limit=plan.final_limit,
        )

    raise UnsupportedQueryStage("QueryPlan could not be translated for Qdrant")


def validate_qdrant_search_request(
    query: SearchQuery,
    *,
    dense_dimensions: int,
) -> str:
    namespace = query.namespace.strip()
    if not namespace:
        raise ValueError("namespace is required for search")
    if _query_requires_dense_vector(query):
        if not query.dense_vector:
            raise ValueError("qdrant dense query vector is required for dense query plans")
        validate_query_dense_dimensions(
            query.dense_vector,
            dense_dimensions=dense_dimensions,
            provider_name=QDRANT_VECTOR_STORE_PROVIDER_SPEC.name,
        )
    if query.query_plan is not None:
        validate_query_plan_capabilities(
            query.query_plan,
            capabilities=QDRANT_VECTOR_STORE_CAPABILITY_SPEC.query_plan,
            provider_name=QDRANT_VECTOR_STORE_PROVIDER_SPEC.name,
        )
        validate_qdrant_query_plan_preflight(query.query_plan)
    return namespace


def validate_qdrant_delete_filter(filter_values: DeleteFilter) -> str:
    namespace = (filter_values.namespace or "").strip()
    if not namespace:
        raise ValueError("namespace is required for delete")
    return namespace


def validate_qdrant_query_plan_preflight(plan: QueryPlan) -> None:
    validate_qdrant_query_plan_shape(plan)
    for prefetch in plan.prefetches:
        _validate_qdrant_prefetch_channels(prefetch)
    prefetch_count = len(plan.prefetches)
    if plan.fuse is not None and prefetch_count < 2:
        raise UnsupportedQueryStage("PrefetchFusion requires at least two prefetches")
    if plan.fuse is not None and plan.fuse.kind == FUSION_KIND_WEIGHTED_RRF:
        if len(plan.fuse.weights) != prefetch_count:
            raise UnsupportedQueryStage(
                "PrefetchFusion(weighted_rrf) requires one weight per prefetch "
                f"(got {len(plan.fuse.weights)} weights for {prefetch_count} prefetches)"
            )


def _validate_qdrant_prefetch_channels(prefetch: Prefetch) -> None:
    if isinstance(prefetch.channel, DenseChannel) and prefetch.channel.vector_field not in {
        "",
        _DENSE_VECTOR_NAME,
    }:
        raise UnsupportedQueryStage(
            "Qdrant adapter supports only the primary dense vector channel"
        )
    for nested in prefetch.nested:
        _validate_qdrant_prefetch_channels(nested)


def _query_requires_dense_vector(query: SearchQuery) -> bool:
    plan = query.query_plan
    if plan is None:
        return True
    return _plan_uses_dense(plan)


def _plan_uses_dense(plan: QueryPlan) -> bool:
    if plan.rerank is not None:
        return True
    return any(_prefetch_uses_dense(prefetch) for prefetch in plan.prefetches)


def _prefetch_uses_dense(prefetch: Prefetch) -> bool:
    if isinstance(prefetch.channel, DenseChannel):
        return True
    return any(_prefetch_uses_dense(nested) for nested in prefetch.nested)


async def search_qdrant_points(
    *,
    client: AsyncQdrantClient,
    collection_name: str,
    query: SearchQuery,
    namespace: str,
    policy: VectorStorePolicy,
    available_sparse_vector_names: frozenset[str] | set[str],
) -> list[SearchResult]:
    if query.has_empty_allowlist():
        return []
    qdrant_filter = build_search_filter(
        query=query,
        namespace=namespace,
        policy=policy,
    )
    plan = query.query_plan or _default_query_plan_for_available_sparse_channels(
        query=query,
        result_limit=query.limit,
        available_sparse_vector_names=available_sparse_vector_names,
    )
    if plan is None:
        raise RuntimeError("Qdrant query-plan capabilities did not produce a plan")
    translated = translate_query_plan(
        plan,
        query=query,
        qdrant_filter=qdrant_filter,
        available_sparse_names=available_sparse_vector_names,
    )
    response = await client.query_points(
        collection_name=collection_name,
        prefetch=translated.prefetch or None,
        query=translated.query,
        using=translated.using,
        query_filter=translated.query_filter,
        limit=translated.limit,
        with_payload=True,
    )
    return [_point_to_result(point, policy=policy) for point in response.points]


def _default_query_plan_for_available_sparse_channels(
    *,
    query: SearchQuery,
    result_limit: int,
    available_sparse_vector_names: frozenset[str] | set[str],
) -> QueryPlan:
    query_sparse_names = set(query.all_sparse_vectors())
    return qdrant_default_query_plan_for_sparse_channels(
        result_limit=result_limit,
        sparse_channels=(
            name for name in available_sparse_vector_names if name in query_sparse_names
        ),
    )


def qdrant_default_query_plan_for_sparse_channels(
    *,
    result_limit: int,
    sparse_channels: Iterable[str],
) -> QueryPlan:
    known_sparse_channels = tuple(
        sorted(name for name in sparse_channels if name in _KNOWN_SPARSE_VECTOR_NAMES)
    )
    if _PRIMARY_SPARSE_VECTOR_NAME in known_sparse_channels:
        return search_profile(DEFAULT_SEARCH_PROFILE, limit=result_limit)
    if known_sparse_channels:
        return default_query_plan(
            result_limit=result_limit,
            sparse_channels=known_sparse_channels,
        )
    return query_plan_preset(QUERY_PLAN_PRESET_DENSE_ONLY, limit=result_limit)
