"""Semantic retrieval eval: real local embeddings on a small fixed corpus."""

from __future__ import annotations

import asyncio
import json
import math
import os
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Literal, cast

import pytest

from rag_core import Engine, Config
from rag_core.config import (
    LOCAL_EMBEDDING_DIMENSIONS,
    LOCAL_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_PROVIDER,
)
from rag_core.evals import EvalResult, load_cases, run_eval
from rag_core.events.types import AuditContext
from rag_core.retrieval_defaults import (
    DEFAULT_RERANK,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_USE_LEXICAL_SEARCH,
)
from rag_core.search import Filter, QueryPlan, RerankBudget, SearchResult

pytestmark = [pytest.mark.eval]

_FIXTURE_DIR = Path(__file__).resolve().parent / "semantic_corpus"
_CORPUS_PATH = _FIXTURE_DIR / "corpus.jsonl"
_CASES_PATH = _FIXTURE_DIR / "cases.jsonl"
_NAMESPACE = "semantic_eval"
_COLLECTION = "docs"
_MAX_SECONDS = 120.0
_HEALTHY_RECALL_AT_5_CEILING = 0.97
_HEALTHY_MRR_CEILING = 0.95
_MIN_DEGRADATION_MARGIN = 0.05
_UNRELATED_QUERY = "remote payroll policy for office laptop reimbursement approvals"

DegradationMode = Literal["rank_shuffle", "wrong_query"]


@dataclass(frozen=True)
class CorpusDoc:
    document_id: str
    markdown: str


@dataclass(frozen=True)
class AggregateMetrics:
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float


_FLOORS = AggregateMetrics(
    recall_at_5=0.95,
    recall_at_10=0.95,
    mrr=0.83,
    ndcg_at_10=0.85,
)


def test_semantic_corpus_fixture_shape() -> None:
    corpus = _load_corpus(_CORPUS_PATH)
    cases = load_cases(_CASES_PATH)
    collections = {doc.document_id for doc in corpus}

    assert 24 <= len(corpus) <= 32
    assert 12 <= len(cases) <= 16
    assert len(collections) == len(corpus)
    assert all(case.namespace == _NAMESPACE for case in cases)
    assert all(case.collections == (_COLLECTION,) for case in cases)
    assert all(set(case.expected_ids) <= collections for case in cases)
    assert any(len(case.expected_ids) > 1 for case in cases)


def test_local_semantic_eval_holds_regression_floors() -> None:
    started = perf_counter()
    results = _run_or_skip()
    elapsed = perf_counter() - started

    assert elapsed <= _MAX_SECONDS, (
        f"local semantic eval took {elapsed:.1f}s; keep the fixed corpus under "
        f"{_MAX_SECONDS:.0f}s post-download"
    )
    _assert_semantic_floors(results)
    _assert_healthy_metrics_are_non_trivial(results)


def test_local_semantic_eval_rejects_degraded_rankings() -> None:
    for degradation in ("rank_shuffle", "wrong_query"):
        started = perf_counter()
        results = _run_or_skip(degradation=degradation)
        elapsed = perf_counter() - started

        assert elapsed <= _MAX_SECONDS, (
            f"{degradation} semantic eval took {elapsed:.1f}s; keep degradation "
            f"proofs under {_MAX_SECONDS:.0f}s post-download"
        )
        _assert_no_search_errors(results)
        failures = _floor_failures(_aggregate(results))
        assert failures, (
            f"{degradation} degradation still met semantic floors "
            f"{_format_metrics(_aggregate(results))}; harden the corpus or "
            "recalibrate floors before accepting new defaults"
        )
        if degradation == "rank_shuffle":
            _assert_shuffle_margin(_aggregate(results))


def test_local_semantic_eval_skip_fails_in_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("RAG_CORE_SKIP_FASTEMBED_DOWNLOAD", "1")

    with pytest.raises(BaseException) as exc_info:
        _run_or_skip()

    assert type(exc_info.value).__name__ == "Failed"
    assert "CI must run the local semantic eval gate" in str(exc_info.value)


def _run_or_skip(degradation: DegradationMode | None = None) -> list[EvalResult]:
    if os.environ.get("RAG_CORE_SKIP_FASTEMBED_DOWNLOAD") == "1":
        _skip_or_fail_unavailable_semantic_gate(
            "RAG_CORE_SKIP_FASTEMBED_DOWNLOAD=1 skips FastEmbed model download"
        )
    try:
        return asyncio.run(_run_semantic_eval(degradation=degradation))
    except Exception as exc:
        if _is_fastembed_download_failure(exc):
            _skip_or_fail_unavailable_semantic_gate(
                "FastEmbed local semantic eval requires one-time model download/cache "
                f"access: {exc}"
            )
        raise


def _skip_or_fail_unavailable_semantic_gate(reason: str) -> None:
    if _truthy_env("CI"):
        pytest.fail(f"CI must run the local semantic eval gate, not skip it: {reason}")
    pytest.skip(reason)


def _truthy_env(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.lower() not in {"", "0", "false", "no", "off"}


async def _run_semantic_eval(
    degradation: DegradationMode | None = None,
) -> list[EvalResult]:
    config = _semantic_config()
    corpus = _load_corpus(_CORPUS_PATH)
    async with Engine(config) as core:
        _assert_local_embedding_runtime(core)
        for doc in corpus:
            await core.add_bytes(
                file_bytes=doc.markdown.encode("utf-8"),
                filename=f"{doc.document_id}.md",
                mime_type="text/markdown",
                namespace=_NAMESPACE,
                collection=_COLLECTION,
                document_id=doc.document_id,
                document_key=f"{doc.document_id}.md",
            )
        eval_core = core
        if degradation is not None:
            eval_core = cast(Engine, _DegradedSearchCore(core, degradation))
        return await run_eval(
            eval_core,
            load_cases(_CASES_PATH),
            k_values=(5, 10, len(corpus)),
            rerank=False,
        )


class _DegradedSearchCore:
    def __init__(self, core: Engine, degradation: DegradationMode) -> None:
        self._core = core
        self._degradation = degradation

    async def search(
        self,
        *,
        query: str,
        namespace: str,
        collections: list[str],
        limit: int = DEFAULT_SEARCH_LIMIT,
        content_types: list[str] | None = None,
        document_ids: list[str] | None = None,
        rerank: bool = DEFAULT_RERANK,
        use_lexical_search: bool = DEFAULT_USE_LEXICAL_SEARCH,
        query_plan: QueryPlan | None = None,
        metadata_filter: Filter | None = None,
        rerank_budget: RerankBudget | None = None,
        audit_context: AuditContext | None = None,
    ) -> list[SearchResult]:
        search_query = _UNRELATED_QUERY if self._degradation == "wrong_query" else query
        hits = await self._core.search(
            query=search_query,
            namespace=namespace,
            collections=collections,
            limit=limit,
            content_types=content_types,
            document_ids=document_ids,
            rerank=rerank,
            use_lexical_search=use_lexical_search,
            query_plan=query_plan,
            metadata_filter=metadata_filter,
            rerank_budget=rerank_budget,
            audit_context=audit_context,
        )
        if self._degradation == "rank_shuffle":
            return list(reversed(hits))
        return hits


def _semantic_config() -> Config:
    config = Config.local()
    return replace(
        config,
        qdrant=replace(
            config.qdrant,
            store_collection=f"rag_core_semantic_eval_{uuid.uuid4().hex}",
        ),
    )


def _assert_local_embedding_runtime(core: Engine) -> None:
    embedding = core.describe_runtime()["embedding"]
    assert embedding == {
        "provider": LOCAL_EMBEDDING_PROVIDER,
        "model": LOCAL_EMBEDDING_MODEL,
        "dimensions": LOCAL_EMBEDDING_DIMENSIONS,
    }, (
        "Local semantic eval model changed; rerun 5x calibration, degradation "
        f"demos, and update floors before accepting new defaults. got={embedding!r}"
    )


def _load_corpus(path: Path) -> list[CorpusDoc]:
    docs: list[CorpusDoc] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            row = json.loads(raw)
            docs.append(
                CorpusDoc(
                    document_id=str(row["document_id"]),
                    markdown=f"# {row['title']}\n\n{row['body']}",
                )
            )
    return docs


def _aggregate(results: Sequence[EvalResult]) -> AggregateMetrics:
    assert results
    return AggregateMetrics(
        recall_at_5=mean(result.recall_at_5 for result in results),
        recall_at_10=mean(result.recall_at_10 for result in results),
        mrr=mean(result.mrr for result in results),
        ndcg_at_10=mean(result.ndcg_at_10 for result in results),
    )


def _assert_semantic_floors(
    results: list[EvalResult],
) -> None:
    _assert_no_search_errors(results)

    metrics = _aggregate(results)
    for (metric_name, actual), (_, floor) in zip(
        _metric_pairs(metrics),
        _metric_pairs(_FLOORS),
        strict=True,
    ):
        assert actual >= floor, _floor_message(metric_name, actual, floor)


def _assert_no_search_errors(results: Sequence[EvalResult]) -> None:
    errors = {
        result.case.case_id or result.case.query: result.error_type
        for result in results
        if result.error_type is not None
    }
    assert not errors, f"semantic eval search errors: {errors}"


def _assert_healthy_metrics_are_non_trivial(results: Sequence[EvalResult]) -> None:
    metrics = _aggregate(results)
    imperfect_count = _imperfect_case_count(results)
    required = math.ceil(len(results) / 3)
    assert imperfect_count >= required, (
        f"semantic corpus has only {imperfect_count}/{len(results)} imperfect cases; "
        "harden near-miss cases and rerun 5x calibration before updating floors"
    )
    assert (
        metrics.recall_at_5 <= _HEALTHY_RECALL_AT_5_CEILING
        or metrics.mrr <= _HEALTHY_MRR_CEILING
    ), (
        "semantic corpus became trivially separable "
        f"{_format_metrics(metrics)}; harden near-miss cases and rerun 5x "
        "calibration before updating floors"
    )


def _imperfect_case_count(results: Sequence[EvalResult]) -> int:
    return sum(
        1
        for result in results
        if (
            result.recall_at_5 < 1.0
            or result.mrr < 1.0
            or result.ndcg_at_10 < 1.0
        )
    )


def _floor_failures(metrics: AggregateMetrics) -> tuple[str, ...]:
    failures: list[str] = []
    for (metric_name, actual), (_, floor) in zip(
        _metric_pairs(metrics),
        _metric_pairs(_FLOORS),
        strict=True,
    ):
        if actual < floor:
            failures.append(f"{metric_name}={actual:.3f}<floor={floor:.3f}")
    return tuple(failures)


def _assert_shuffle_margin(metrics: AggregateMetrics) -> None:
    for (metric_name, degraded), (_, floor) in zip(
        _metric_pairs(metrics),
        _metric_pairs(_FLOORS),
        strict=True,
    ):
        assert floor >= degraded + _MIN_DEGRADATION_MARGIN, (
            f"semantic eval {metric_name} floor={floor:.3f} must stay at least "
            f"{_MIN_DEGRADATION_MARGIN:.2f} above rank-shuffle degradation "
            f"{degraded:.3f}; harden the corpus or recalibrate floors"
        )


def _metric_pairs(metrics: AggregateMetrics) -> tuple[tuple[str, float], ...]:
    return (
        ("recall@5", metrics.recall_at_5),
        ("recall@10", metrics.recall_at_10),
        ("mrr", metrics.mrr),
        ("ndcg@10", metrics.ndcg_at_10),
    )


def _format_metrics(metrics: AggregateMetrics) -> str:
    return " ".join(
        f"{metric_name}={value:.3f}" for metric_name, value in _metric_pairs(metrics)
    )


def _floor_message(metric: str, actual: float, floor: float) -> str:
    return (
        f"semantic eval {metric}={actual:.3f} below floor={floor:.3f}; "
        "inspect retrieval changes, then rerun 5x calibration and degradation demos "
        "before updating floors"
    )


def _is_fastembed_download_failure(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "fastembed local embedding model failed to load" in text
        or "huggingface" in text
        or "model download" in text
        or "could not download" in text
    )
