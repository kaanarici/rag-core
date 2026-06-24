from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, Union

RangeBound: TypeAlias = float | str


@dataclass(frozen=True)
class Term:
    """Exact-match against a payload field."""

    field: str
    value: str | int | float | bool


@dataclass(frozen=True)
class In:
    """Match when a payload field's value is in the given set."""

    field: str
    values: tuple[str | int | float, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("In.values must not be empty")


@dataclass(frozen=True)
class Range:
    """Numeric or ISO-string-comparable range over a payload field.

    At least one of ``gte``/``lte``/``gt``/``lt`` must be set; an entirely
    open range is rejected because it filters nothing and is almost always
    a programming bug at the call site.
    """

    field: str
    gte: RangeBound | None = None
    lte: RangeBound | None = None
    gt: RangeBound | None = None
    lt: RangeBound | None = None

    def __post_init__(self) -> None:
        if (
            self.gte is None
            and self.lte is None
            and self.gt is None
            and self.lt is None
        ):
            raise ValueError("Range requires at least one of gte/lte/gt/lt")


@dataclass(frozen=True)
class Geo:
    """Geo-radius match around ``(lat, lon)`` using ``radius_m`` metres."""

    field: str
    lat: float
    lon: float
    radius_m: float

    def __post_init__(self) -> None:
        if not -90.0 <= self.lat <= 90.0:
            raise ValueError("Geo.lat must be in [-90, 90]")
        if not -180.0 <= self.lon <= 180.0:
            raise ValueError("Geo.lon must be in [-180, 180]")
        if self.radius_m <= 0.0:
            raise ValueError("Geo.radius_m must be positive")


@dataclass(frozen=True)
class And:
    """Logical AND over child filters. Empty conjunction is rejected."""

    filters: tuple["Filter", ...]

    def __post_init__(self) -> None:
        if not self.filters:
            raise ValueError("And.filters must not be empty")


@dataclass(frozen=True)
class Or:
    """Logical OR over child filters. Empty disjunction is rejected."""

    filters: tuple["Filter", ...]

    def __post_init__(self) -> None:
        if not self.filters:
            raise ValueError("Or.filters must not be empty")


@dataclass(frozen=True)
class Not:
    """Negation of a single child filter."""

    filter: "Filter"


Filter: TypeAlias = Union[Term, In, Range, Geo, And, Or, Not]
"""Discriminated union of payload-filter AST nodes.

Lives alongside the first-class ``namespace``/``collections``/``document_ids``/
``content_types`` filters on ``SearchQuery``: those stay top-level, this AST
is for everything else (date ranges, categories, languages, geo, boolean
composition) without forking a ``VectorStore``.
"""


__all__ = [
    "And",
    "Filter",
    "Geo",
    "In",
    "Not",
    "Or",
    "Range",
    "Term",
]
