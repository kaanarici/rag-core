from __future__ import annotations

import math
from dataclasses import replace

from rag_core.search.vector_models import SearchResult


def finite_score_or_zero(score: object) -> float:
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return 0.0
    try:
        value = float(score)
    except (OverflowError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def sanitize_result_score(result: SearchResult) -> SearchResult:
    score = finite_score_or_zero(result.score)
    if isinstance(result.score, float) and score == result.score:
        return result
    return replace(result, score=score)
