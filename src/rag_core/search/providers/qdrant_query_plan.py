"""Translate the typed :class:`QueryPlan` AST to Qdrant Query API arguments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from qdrant_client import models as rest

from rag_core.search.query_plan import (
    BOOST_KIND_EXP_DECAY,
    BOOST_KIND_GAUSS_DECAY,
    BOOST_KIND_LINEAR_DECAY,
    BOOST_KIND_RAW,
    Boost,
    DEFAULT_RRF_K,
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
from rag_core.search.request_models import SearchQuery

from .qdrant_query_channels import (
    ensure_qdrant_sparse_channel_supported,
    query_for_qdrant_channel,
    using_for_qdrant_channel,
)
from .qdrant_shared import _KNOWN_SPARSE_VECTOR_NAMES


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
