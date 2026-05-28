from __future__ import annotations

from typing import Any, cast

from qdrant_client import models as rest

from rag_core.search.filters import (
    And,
    Filter,
    Geo,
    In,
    Not,
    Or,
    Range,
    Term,
)
from rag_core.search.query_plan import UnsupportedQueryStage


def metadata_filter_to_qdrant(filter: Filter) -> rest.Filter:
    """Translate the engine's ``Filter`` AST into a Qdrant ``rest.Filter``."""

    return rest.Filter(must=cast(Any, [_translate_filter_node(filter)]))


def _translate_filter_node(node: Filter) -> object:
    if isinstance(node, Term):
        return rest.FieldCondition(
            key=node.field,
            match=rest.MatchValue(value=cast(Any, node.value)),
        )
    if isinstance(node, In):
        return rest.FieldCondition(
            key=node.field,
            match=rest.MatchAny(any=cast(Any, list(node.values))),
        )
    if isinstance(node, Range):
        return rest.FieldCondition(
            key=node.field,
            range=rest.Range(
                gte=_qdrant_range_bound(node.gte),
                lte=_qdrant_range_bound(node.lte),
                gt=_qdrant_range_bound(node.gt),
                lt=_qdrant_range_bound(node.lt),
            ),
        )
    if isinstance(node, Geo):
        return rest.FieldCondition(
            key=node.field,
            geo_radius=rest.GeoRadius(
                center=rest.GeoPoint(lat=node.lat, lon=node.lon),
                radius=node.radius_m,
            ),
        )
    if isinstance(node, And):
        return rest.Filter(
            must=cast(Any, [_translate_filter_node(child) for child in node.filters])
        )
    if isinstance(node, Or):
        return rest.Filter(
            should=cast(Any, [_translate_filter_node(child) for child in node.filters])
        )
    if isinstance(node, Not):
        return rest.Filter(must_not=cast(Any, [_translate_filter_node(node.filter)]))
    raise UnsupportedQueryStage(
        f"qdrant adapter cannot translate Filter node {type(node).__name__}"
    )


def _qdrant_range_bound(bound: float | str | None) -> float | None:
    if bound is None:
        return None
    if isinstance(bound, str):
        raise UnsupportedQueryStage(
            "qdrant adapter cannot translate string Range filters"
        )
    return bound


__all__ = ["metadata_filter_to_qdrant"]
