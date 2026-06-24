"""Eval result models and a runner for ``core.search``.

The runner consumes :class:`rag_core.Engine` through its public surface
only. An eval case feeds the engine a query and the engine yields
:class:`SearchResult` objects, from which we extract a stable id per result
and feed the metric functions.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re
from time import perf_counter
from typing import TYPE_CHECKING, Any

from rag_core.evals.cases import EvalCase, load_cases
from rag_core.evals.metrics import mrr, ndcg_at_k, recall_at_k
from rag_core.retrieval_defaults import DEFAULT_RERANK, DEFAULT_SEARCH_LIMIT
from rag_core.search.context_pack import build_context_pack

if TYPE_CHECKING:
    from rag_core import Engine
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
    context_recall: float = 0.0
    citation_count: int = 0
    source_count: int = 0
    forbidden_leak_count: int = 0
    context_token_estimate: int = 0
    context_char_count: int = 0
    context_contains_pass: bool = True
    prompt_safety_pass: bool = True


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


def _context_metrics(case: EvalCase, hits: Sequence[SearchResult]) -> dict[str, Any]:
    if not _has_context_expectations(case):
        return {}
    pack = build_context_pack(
        hits,
        query=case.query,
        max_chars=case.max_context_chars,
        max_tokens=case.max_context_tokens,
    )
    text = pack.as_prompt_text()
    lowered = text.lower()
    expected = tuple(item.lower() for item in case.expected_context_contains)
    forbidden = tuple(
        item.lower()
        for item in (
            *case.forbidden_context_contains,
            *case.forbidden_private_identifiers,
        )
    )
    matched = sum(1 for item in expected if item in lowered)
    forbidden_leak_count = sum(1 for item in forbidden if item in lowered)
    context_contains_pass = matched == len(expected)
    citation_count = len(pack.citations)
    source_count = len(pack.source_previews)
    prompt_safety_pass = (
        forbidden_leak_count == 0
        and citation_count >= case.expected_citation_count_min
        and source_count >= case.expected_source_count_min
        and (
            case.max_context_chars is None
            or len(text) <= case.max_context_chars
        )
        and (
            case.max_context_tokens is None
            or pack.token_estimate <= case.max_context_tokens
        )
    )
    return {
        "context_recall": matched / len(expected) if expected else 1.0,
        "citation_count": citation_count,
        "source_count": source_count,
        "forbidden_leak_count": forbidden_leak_count,
        "context_token_estimate": pack.token_estimate,
        "context_char_count": len(text),
        "context_contains_pass": context_contains_pass,
        "prompt_safety_pass": prompt_safety_pass,
    }


def _has_context_expectations(case: EvalCase) -> bool:
    return bool(
        case.expected_context_contains
        or case.forbidden_context_contains
        or case.forbidden_private_identifiers
        or case.expected_citation_count_min
        or case.expected_source_count_min
        or case.max_context_chars is not None
        or case.max_context_tokens is not None
    )


_SAFE_ERROR_TYPE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,79}")


def _safe_error_type(exc: Exception) -> str:
    error_type = type(exc).__name__
    if _SAFE_ERROR_TYPE_RE.fullmatch(error_type):
        return error_type
    return "Exception"


async def run_eval(
    core: Engine,
    cases: Sequence[EvalCase],
    k_values: Sequence[int] = (5, 10),
    *,
    rerank: bool = DEFAULT_RERANK,
    rerank_budget: RerankBudget | None = None,
    query_plan: QueryPlan | None = None,
    max_concurrency: int = 1,
) -> list[EvalResult]:
    """Run each case through ``core.search`` and compute metrics.

    The runner does not mutate ``core``. ``k_values`` controls only the
    retrieval limit (``max(k_values)``); the returned ``EvalResult`` always
    reports recall at 5 and 10 plus nDCG@10 so aggregation is uniform.
    Set ``rerank=True`` to evaluate the configured reranker against the
    same labelled cases.
    """
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")
    if not cases:
        return []
    limit = max(*k_values, DEFAULT_SEARCH_LIMIT) if k_values else DEFAULT_SEARCH_LIMIT
    if max_concurrency == 1:
        return [
            await _run_eval_case(
                core,
                case,
                limit=limit,
                rerank=rerank,
                rerank_budget=rerank_budget,
                query_plan=query_plan,
            )
            for case in cases
        ]

    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_bounded(case: EvalCase) -> EvalResult:
        async with semaphore:
            return await _run_eval_case(
                core,
                case,
                limit=limit,
                rerank=rerank,
                rerank_budget=rerank_budget,
                query_plan=query_plan,
            )

    return list(await asyncio.gather(*(run_bounded(case) for case in cases)))


async def _run_eval_case(
    core: Engine,
    case: EvalCase,
    *,
    limit: int,
    rerank: bool,
    rerank_budget: RerankBudget | None,
    query_plan: QueryPlan | None,
) -> EvalResult:
    started = perf_counter()
    try:
        hits = await core.search(
            query=case.query,
            namespace=case.namespace,
            collections=list(case.collections),
            limit=limit,
            rerank=rerank,
            rerank_budget=rerank_budget,
            query_plan=query_plan,
        )
    except Exception as exc:  # noqa: BLE001 - eval reports failure per case.
        latency_ms = (perf_counter() - started) * 1000.0
        return EvalResult(
            case=case,
            retrieved_ids=(),
            recall_at_5=0.0,
            recall_at_10=0.0,
            mrr=0.0,
            ndcg_at_10=0.0,
            latency_ms=latency_ms,
            error_type=_safe_error_type(exc),
        )
    latency_ms = (perf_counter() - started) * 1000.0
    expected_ids = set(case.expected_ids)
    grade_ids = set(case.expected_grades or {})
    retrieved_ids = tuple(
        _result_id(hit, expected_ids=expected_ids, grade_ids=grade_ids)
        for hit in hits
    )
    relevant = case.expected_ids
    grades = _grades_for(case)
    return EvalResult(
        case=case,
        retrieved_ids=retrieved_ids,
        recall_at_5=recall_at_k(retrieved_ids, relevant, 5),
        recall_at_10=recall_at_k(retrieved_ids, relevant, 10),
        mrr=mrr(retrieved_ids, relevant),
        ndcg_at_10=ndcg_at_k(retrieved_ids, grades, 10),
        latency_ms=latency_ms,
        **_context_metrics(case, hits),
    )

__all__ = ["EvalCase", "EvalResult", "load_cases", "run_eval"]
