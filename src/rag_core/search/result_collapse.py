"""Deterministic exact-content collapse shared by evidence and context retrieval."""

from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
import math
from rag_core.search.vector_models import SearchResult, SEARCH_RESULT_TYPE_TEXT
from rag_core.search.context_pack_sources import (
    source_reference_from_result,
    source_locator_from_result,
)


@dataclass(frozen=True)
class CollapsedSearchResult:
    primary: SearchResult
    equivalents: tuple[SearchResult, ...]


def collapse_search_results(
    hits: Sequence[SearchResult],
    *,
    limit: int,
) -> tuple[CollapsedSearchResult, ...]:
    groups: dict[tuple[str, str], list[tuple[int, SearchResult]]] = {}
    for rank, hit in enumerate(hits):
        result_type = hit.result_type or SEARCH_RESULT_TYPE_TEXT
        groups.setdefault((result_type, hit.text), []).append((rank, hit))

    collapsed: list[tuple[int, tuple[str, ...], CollapsedSearchResult]] = []
    for members in groups.values():
        ordered_hits = sorted(
            (hit for _, hit in members),
            key=_representative_sort_key,
        )
        collapsed.append(
            (
                min(rank for rank, _ in members),
                _stable_source_key(ordered_hits[0]),
                CollapsedSearchResult(
                    primary=ordered_hits[0],
                    equivalents=tuple(ordered_hits[1:]),
                ),
            )
        )
    collapsed.sort(key=lambda item: (item[0], item[1]))
    return tuple(item[2] for item in collapsed[:limit])


def _representative_sort_key(hit: SearchResult) -> tuple[float, tuple[str, ...]]:
    score = hit.score if math.isfinite(hit.score) else float("-inf")
    return (-score, _stable_source_key(hit))


def _stable_source_key(hit: SearchResult) -> tuple[str, ...]:
    source = source_reference_from_result(hit)
    locator = source_locator_from_result(hit)
    return (
        source.source_id,
        hit.document_id or "",
        hit.document_key or "",
        str(locator.chunk_index if locator.chunk_index is not None else ""),
        str(locator.page_number if locator.page_number is not None else ""),
        str(locator.line_start if locator.line_start is not None else ""),
        str(locator.start_offset if locator.start_offset is not None else ""),
        hit.id,
    )
