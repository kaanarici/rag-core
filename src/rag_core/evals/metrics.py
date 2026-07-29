"""Pure retrieval-quality metric functions.

All functions are deterministic, allocation-light, and total: they return
``0.0`` for empty input or non-positive ``k`` rather than raising. That
makes them safe to call inside a runner that may have empty cases or be
mid-bring-up.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int,
) -> float:
    """Fraction of relevant ids that appear in the first ``k`` retrieved ids.

    Returns 0.0 if ``retrieved_ids`` is empty, ``k`` is non-positive, or
    ``relevant_ids`` is empty.
    """
    if k <= 0 or not retrieved_ids:
        return 0.0
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    seen = {item for item in retrieved_ids[:k] if item in relevant}
    return len(seen) / len(relevant)


def mrr(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
) -> float:
    """Mean reciprocal rank for one query: 1 / rank of the first relevant hit.

    Returns 0.0 if ``retrieved_ids`` is empty, ``relevant_ids`` is empty,
    or no retrieved id is relevant.
    """
    if not retrieved_ids:
        return 0.0
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    for index, item in enumerate(retrieved_ids, start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevant_grades: Mapping[str, int],
    k: int,
) -> float:
    """Normalized discounted cumulative gain at ``k`` with graded relevance.

    Grades are non-negative integers. Items missing from ``relevant_grades``
    contribute zero gain. Returns 0.0 if ``retrieved_ids`` is empty,
    ``k`` is non-positive, or no item has positive grade.
    """
    if k <= 0 or not retrieved_ids:
        return 0.0
    positive_grades = sorted(
        (grade for grade in relevant_grades.values() if grade > 0),
        reverse=True,
    )
    if not positive_grades:
        return 0.0
    dcg = 0.0
    seen_gain_ids: set[str] = set()
    for index, item in enumerate(retrieved_ids[:k], start=1):
        if item in seen_gain_ids:
            continue
        grade = relevant_grades.get(item, 0)
        if grade > 0:
            dcg += (2**grade - 1) / math.log2(index + 1)
            seen_gain_ids.add(item)
    idcg = 0.0
    for index, grade in enumerate(positive_grades[:k], start=1):
        idcg += (2**grade - 1) / math.log2(index + 1)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


__all__ = ["mrr", "ndcg_at_k", "recall_at_k"]
