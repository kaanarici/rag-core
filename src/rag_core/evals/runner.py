"""Eval result models and a runner for ``core.search``.

The runner consumes :class:`rag_core.RAGCore` through its public surface
only. An eval case feeds the engine a query and the engine yields
:class:`SearchResult` objects, from which we extract a stable id per result
and feed the metric functions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re
from time import perf_counter
from typing import TYPE_CHECKING

from rag_core.evals.cases import EvalCase, load_cases
from rag_core.evals.metrics import mrr, ndcg_at_k, recall_at_k
from rag_core.retrieval_defaults import DEFAULT_RERANK, DEFAULT_SEARCH_LIMIT

if TYPE_CHECKING:
    from rag_core import RAGCore
    from rag_core.search import QueryPlan, RerankBudget, SearchResult

@dataclass(frozen=True)
class EvalResult:
    """Computed metrics for a single :class:`EvalCase`."""

    case: EvalCase
    retrieved_ids: tuple[str, ...]
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    latency_ms: float
    error_type: str | None = None


def _result_id(
    result: SearchResult,
    *,
    expected_ids: set[str],
    grade_ids: set[str],
) -> str:
    if result.id in expected_ids or result.id in grade_ids:
        return result.id
    if result.document_id and (
        result.document_id in expected_ids or result.document_id in grade_ids
    ):
        return result.document_id
    if result.document_key and (
        result.document_key in expected_ids or result.document_key in grade_ids
    ):
        return result.document_key
    return result.document_id or result.id


def _grades_for(case: EvalCase) -> Mapping[str, int]:
    if case.expected_grades is not None:
        return case.expected_grades
    return {expected_id: 1 for expected_id in case.expected_ids}


_SAFE_ERROR_TYPE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,79}")


def _safe_error_type(exc: Exception) -> str:
    error_type = type(exc).__name__
    if _SAFE_ERROR_TYPE_RE.fullmatch(error_type):
        return error_type
    return "Exception"


async def run_eval(
    core: RAGCore,
    cases: Sequence[EvalCase],
    k_values: Sequence[int] = (5, 10),
    *,
    rerank: bool = DEFAULT_RERANK,
    rerank_budget: RerankBudget | None = None,
    query_plan: QueryPlan | None = None,
) -> list[EvalResult]:
    """Run each case through ``core.search`` and compute metrics.

    The runner does not mutate ``core``. ``k_values`` controls only the
    retrieval limit (``max(k_values)``); the returned ``EvalResult`` always
    reports recall at 5 and 10 plus nDCG@10 so aggregation is uniform.
    Set ``rerank=True`` to evaluate the configured reranker against the
    same labelled cases.
    """
    if not cases:
        return []
    limit = max(*k_values, DEFAULT_SEARCH_LIMIT) if k_values else DEFAULT_SEARCH_LIMIT
    results: list[EvalResult] = []
    for case in cases:
        started = perf_counter()
        try:
            hits = await core.search(
                query=case.query,
                namespace=case.namespace,
                corpus_ids=list(case.corpus_ids),
                limit=limit,
                rerank=rerank,
                rerank_budget=rerank_budget,
                query_plan=query_plan,
            )
        except Exception as exc:  # noqa: BLE001 - eval reports failure per case.
            latency_ms = (perf_counter() - started) * 1000.0
            results.append(
                EvalResult(
                    case=case,
                    retrieved_ids=(),
                    recall_at_5=0.0,
                    recall_at_10=0.0,
                    mrr=0.0,
                    ndcg_at_10=0.0,
                    latency_ms=latency_ms,
                    error_type=_safe_error_type(exc),
                )
            )
            continue
        latency_ms = (perf_counter() - started) * 1000.0
        expected_ids = set(case.expected_ids)
        grade_ids = set(case.expected_grades or {})
        retrieved_ids = tuple(
            _result_id(hit, expected_ids=expected_ids, grade_ids=grade_ids)
            for hit in hits
        )
        relevant = case.expected_ids
        grades = _grades_for(case)
        results.append(
            EvalResult(
                case=case,
                retrieved_ids=retrieved_ids,
                recall_at_5=recall_at_k(retrieved_ids, relevant, 5),
                recall_at_10=recall_at_k(retrieved_ids, relevant, 10),
                mrr=mrr(retrieved_ids, relevant),
                ndcg_at_10=ndcg_at_k(retrieved_ids, grades, 10),
                latency_ms=latency_ms,
            )
        )
    return results

__all__ = ["EvalCase", "EvalResult", "load_cases", "run_eval"]
