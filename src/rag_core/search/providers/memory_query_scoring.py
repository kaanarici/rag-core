from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from rag_core.search.query_plan import DEFAULT_RRF_K
from rag_core.search.sparse_channels import PRIMARY_SPARSE_CHANNEL
from rag_core.search.vector_models import SparseVector


@dataclass
class MemoryPoint:
    id: str
    dense: list[float]
    sparse: dict[str, SparseVector]
    payload: dict[str, object] = field(default_factory=dict)


def rank_dense_points(
    query_vector: list[float],
    candidates: Sequence[MemoryPoint],
    limit: int,
) -> list[str]:
    if not query_vector:
        return []
    scored: list[tuple[float, str]] = []
    for stored in candidates:
        if not stored.dense:
            continue
        score = _cosine_similarity(query_vector, stored.dense)
        scored.append((score, stored.id))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [point_id for _, point_id in scored[:limit]]


def rank_sparse_points(
    name: str,
    query_vector: SparseVector,
    candidates: Sequence[MemoryPoint],
    limit: int,
    *,
    fallback_to_primary: bool,
) -> list[str]:
    if not query_vector.indices:
        return []
    scored: list[tuple[float, str]] = []
    query_map = dict(zip(query_vector.indices, query_vector.values, strict=True))
    for stored in candidates:
        sparse = stored.sparse.get(name)
        if sparse is None and fallback_to_primary:
            sparse = stored.sparse.get(PRIMARY_SPARSE_CHANNEL)
        if sparse is None:
            continue
        score = _sparse_dot(query_map, sparse)
        if score == 0.0:
            continue
        scored.append((score, stored.id))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [point_id for _, point_id in scored[:limit]]


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    limit: int,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, point_id in enumerate(ranking):
            scores[point_id] = scores.get(point_id, 0.0) + 1.0 / (
                DEFAULT_RRF_K + rank + 1
            )
    fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return fused[:limit]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    length = min(len(a), len(b))
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for index in range(length):
        ai = a[index]
        bi = b[index]
        dot += ai * bi
        norm_a += ai * ai
        norm_b += bi * bi
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


def _sparse_dot(query_map: dict[int, float], stored: SparseVector) -> float:
    score = 0.0
    for index, value in zip(stored.indices, stored.values, strict=True):
        score += query_map.get(index, 0.0) * value
    return score
