"""Typed AST for multi-stage vector-store-side queries.

The basic ``VectorStore.search`` contract continues to accept a ``SearchQuery``;
``QueryPlan`` is the optional, richer expression of "do these prefetches, fuse,
then maybe rerank, then maybe rescore." Adapters translate as much of the plan
as they support and raise :class:`UnsupportedQueryStage` for stages they cannot
honor. Callers can either downgrade the plan or pick a different backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Literal, Optional, Union


@dataclass(frozen=True)
class DenseChannel:
    """Dense vector retrieval against the dense channel."""

    vector_field: str = ""
    using_query_vector: str = "primary"


@dataclass(frozen=True)
class SparseChannel:
    """Sparse vector retrieval against a named sparse channel."""

    vector_field: str = "bm25"
    using_query_vector: str = "bm25"


Channel = Union[DenseChannel, SparseChannel]


@dataclass(frozen=True)
class Prefetch:
    """A retrieval stage that produces a candidate list.

    ``nested`` enables multi-stage candidate generation: e.g. retrieve a wide
    sparse shortlist, then narrow with a dense rerank within the shortlist.
    """

    channel: Channel
    limit: int
    nested: tuple["Prefetch", ...] = ()

    def __post_init__(self) -> None:
        _require_positive_int(self.limit, "Prefetch.limit")


FusionKind = Literal["rrf", "dbsf", "weighted_rrf"]


@dataclass(frozen=True)
class PrefetchFusion:
    """Combine multiple prefetch results using a named fusion strategy.

    ``weights`` is meaningful only when ``kind == 'weighted_rrf'``.
    ``rrf_k`` is meaningful only when ``kind in ('rrf', 'weighted_rrf')``;
    default 60 matches the textbook RRF constant.
    """

    kind: FusionKind = "rrf"
    weights: tuple[float, ...] = ()
    rrf_k: int = 60

    def __post_init__(self) -> None:
        _require_int(self.rrf_k, "PrefetchFusion.rrf_k")
        if self.kind == "weighted_rrf" and not self.weights:
            raise ValueError("PrefetchFusion(kind='weighted_rrf') requires weights")
        if self.kind != "weighted_rrf" and self.weights:
            raise ValueError(
                f"PrefetchFusion(kind={self.kind!r}) does not support weights"
            )


@dataclass(frozen=True)
class Mmr:
    """Maximum Marginal Relevance diversity rerank. Optional capability."""

    diversity: float
    limit: int

    def __post_init__(self) -> None:
        if not 0.0 < self.diversity < 1.0:
            raise ValueError("Mmr.diversity must be in the open interval (0, 1)")
        _require_positive_int(self.limit, "Mmr.limit")


@dataclass(frozen=True)
class Boost:
    """Score formula rescore. Optional capability.

    Cross-vendor common shapes (linear/exp/gauss decay over a payload field)
    are first-class via ``kind`` and ``field``. Vendor-specific shapes go in
    ``params``.
    """

    kind: Literal["linear_decay", "exp_decay", "gauss_decay", "raw"]
    field: str = ""
    params: dict[str, object] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind != "raw" and not self.field:
            raise ValueError(f"Boost(kind={self.kind!r}) requires a field")


PlanStage = Union[Prefetch, PrefetchFusion, Mmr, Boost]


@dataclass(frozen=True)
class QueryPlan:
    """A linear sequence of vector-store-side query stages.

    The first stage(s) must be Prefetch(es) producing candidate lists. ``fuse``
    combines them when there is more than one prefetch. Subsequent stages
    refine the merged list. The plan is consumed by ``VectorStore.search``;
    adapters that cannot honor a stage raise :class:`UnsupportedQueryStage`.
    """

    prefetches: tuple[Prefetch, ...]
    fuse: Optional[PrefetchFusion] = None
    rerank: Optional[Mmr] = None
    boost: Optional[Boost] = None
    final_limit: int = 20
    search_profile: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.prefetches:
            raise ValueError("QueryPlan must have at least one Prefetch")
        if self.fuse is not None and len(self.prefetches) < 2:
            raise ValueError("PrefetchFusion stage requires at least two Prefetches")
        if len(self.prefetches) > 1 and self.fuse is None:
            raise ValueError("Multiple prefetches require a PrefetchFusion stage")
        _require_positive_int(self.final_limit, "final_limit")


class UnsupportedQueryStage(Exception):
    """Raised by a VectorStore adapter when it cannot honor a QueryPlan stage."""


def _require_positive_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _require_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
