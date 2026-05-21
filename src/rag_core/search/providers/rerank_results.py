"""Shared provider result validation for rerank adapters."""

from __future__ import annotations

import logging
import math
from typing import TypeGuard

from rag_core.search.types import RerankResult

logger = logging.getLogger(__name__)


class ValidatedRerankResults(list[RerankResult]):
    """Validated rerank rows plus the raw provider row count."""

    def __init__(
        self,
        rows: list[RerankResult],
        *,
        provider_result_count: int,
    ) -> None:
        super().__init__(rows)
        self.provider_result_count = provider_result_count


def rerank_provider_result_count(results: list[RerankResult]) -> int:
    count = getattr(results, "provider_result_count", None)
    if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
        return count
    return len(results)


def _safe_value_type(value: object) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if value is None:
        return "none"
    return "object"


def _is_int_index(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_score_value(value: object) -> TypeGuard[int | float | str]:
    return isinstance(value, (int, float, str)) and not isinstance(value, bool)


def safe_indexed_rerank_results(
    *,
    rows: list[tuple[object, object]],
    documents: list[str],
    provider_name: str,
) -> ValidatedRerankResults:
    results: list[RerankResult] = []
    seen_indices: set[int] = set()
    for raw_index, raw_score in rows:
        if not _is_int_index(raw_index) or not 0 <= raw_index < len(documents):
            logger.warning(
                "%s returned invalid rerank index (value_type=%s)",
                provider_name,
                _safe_value_type(raw_index),
            )
            continue
        if raw_index in seen_indices:
            logger.warning("%s returned duplicate rerank index", provider_name)
            continue
        if not _is_score_value(raw_score):
            logger.warning(
                "%s returned invalid rerank score (reason=invalid_type value_type=%s)",
                provider_name,
                _safe_value_type(raw_score),
            )
            continue
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            logger.warning(
                "%s returned invalid rerank score (reason=invalid_value value_type=%s)",
                provider_name,
                _safe_value_type(raw_score),
            )
            continue
        if not math.isfinite(score):
            logger.warning(
                "%s returned non-finite rerank score (value_type=%s)",
                provider_name,
                _safe_value_type(raw_score),
            )
            continue
        seen_indices.add(raw_index)
        results.append(
            RerankResult(
                index=raw_index,
                score=score,
                text=documents[raw_index],
            )
        )
    results.sort(key=lambda result: result.score, reverse=True)
    return ValidatedRerankResults(results, provider_result_count=len(rows))
