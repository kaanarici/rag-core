"""Unit tests for EvalCase/EvalResult and the runner shape."""

from __future__ import annotations

import asyncio
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

import pytest

import rag_core.evals.runner as runner_module
from rag_core import Engine
from rag_core.evals import (
    EvalCase,
    EvalResult,
    eval_report,
    load_cases,
    run_eval,
)
from rag_core.search.planning import query_plan_preset
from rag_core.search.providers.memory_store import InMemoryVectorStore
from rag_core.search.request_models import (
    RerankBudget,
    RerankResult,
)
from tests.support import (
    BASELINE_VOCABULARY,
    FakeEmbeddingProvider,
    FakeReranker,
    FakeSparseEmbedder,
    KeywordEmbeddingProvider,
    KeywordSparseEmbedder,
    RecordingVectorStore,
    make_search_result,
    make_test_config,
)


def test_eval_case_is_frozen() -> None:
    case = EvalCase(query="q", namespace="ns", collections=("c",), expected_ids=("a",))
    with pytest.raises((FrozenInstanceError, AttributeError)):
        case.query = "other"  # type: ignore[misc]


def test_eval_case_stores_expected_ids() -> None:
    case = EvalCase(
        query="q",
        namespace="ns",
        collections=("c",),
        expected_ids=("doc-a",),
    )

    assert case.expected_ids == ("doc-a",)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        cast(Any, EvalCase)(
            query="q",
            namespace="ns",
            collections=("c",),
            expected_chunk_ids=("doc-a",),
        )


def test_eval_result_is_frozen() -> None:
    case = EvalCase(query="q", namespace="ns", collections=("c",), expected_ids=("a",))
    result = EvalResult(
        case=case,
        retrieved_ids=("a",),
        recall_at_5=1.0,
        recall_at_10=1.0,
        mrr=1.0,
        ndcg_at_10=1.0,
        latency_ms=1.5,
    )
    with pytest.raises((FrozenInstanceError, AttributeError)):
        result.recall_at_5 = 0.0  # type: ignore[misc]


def test_load_cases_round_trip(tmp_path: Path) -> None:
    payload = [
        {
            "query": "q1",
            "namespace": "ns",
            "collections": ["c"],
            "expected_ids": ["x", "y"],
        },
        {
            "query": "q2",
            "namespace": "ns",
            "collections": ["c"],
            "expected_ids": ["z", "w"],
            "expected_grades": {"z": 3, "w": 1},
        },
    ]
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in payload) + "\n", encoding="utf-8")

    cases = load_cases(path)

    assert len(cases) == 2
    assert cases[0] == EvalCase(
        query="q1",
        namespace="ns",
        collections=("c",),
        expected_ids=("x", "y"),
        expected_grades=None,
    )
    assert cases[1].expected_ids == ("z", "w")
    assert cases[1].expected_grades == {"z": 3, "w": 1}


def test_load_cases_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "\n"
        + json.dumps(
            {
                "query": "q",
                "namespace": "ns",
                "collections": ["c"],
                "expected_ids": ["x"],
            }
        )
        + "\n\n",
        encoding="utf-8",
    )

    assert len(load_cases(path)) == 1


def _make_keyword_core(collection: str) -> Engine:
    return Engine(
        make_test_config(
            qdrant_collection=collection,
            embedding_dimensions=len(BASELINE_VOCABULARY),
            qdrant_dimension_aware_collection=False,
        ),
        embedding_provider=KeywordEmbeddingProvider(BASELINE_VOCABULARY),
        sparse_embedder=KeywordSparseEmbedder(BASELINE_VOCABULARY),
        vector_store=InMemoryVectorStore(),
    )


def test_run_eval_empty_cases_returns_empty() -> None:
    async def go() -> list[EvalResult]:
        core = _make_keyword_core("rag_core_eval_smoke")
        try:
            await core.ensure_ready()
            return await run_eval(core, [])
        finally:
            await core.close()

    assert asyncio.run(go()) == []


def test_run_eval_rejects_non_positive_max_concurrency() -> None:
    with pytest.raises(ValueError, match="max_concurrency must be positive"):
        asyncio.run(run_eval(cast(Engine, object()), [], max_concurrency=0))


def test_run_eval_default_matches_explicit_sequential_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StableCore:
        async def search(self, **kwargs: object) -> list[object]:
            return [make_search_result(document_id=cast(str, kwargs["query"]))]

    class _Ticker:
        current = 0.0

        def __call__(self) -> float:
            self.current += 0.001
            return self.current

    cases = [
        EvalCase(
            query="billing",
            namespace="rt",
            collections=("docs",),
            expected_ids=("billing",),
        ),
        EvalCase(
            query="shipping",
            namespace="rt",
            collections=("docs",),
            expected_ids=("shipping",),
        ),
    ]
    ticker = _Ticker()
    monkeypatch.setattr(runner_module, "perf_counter", ticker)

    default_results = asyncio.run(run_eval(cast(Engine, _StableCore()), cases))
    ticker.current = 0.0
    explicit_results = asyncio.run(
        run_eval(cast(Engine, _StableCore()), cases, max_concurrency=1)
    )

    assert default_results == explicit_results


def test_run_eval_concurrent_path_preserves_input_order() -> None:
    class _DelayedCore:
        async def search(self, **kwargs: object) -> list[object]:
            query = cast(str, kwargs["query"])
            await asyncio.sleep(
                {
                    "doc-0": 0.03,
                    "doc-1": 0.005,
                    "doc-2": 0.02,
                    "doc-3": 0.001,
                }[query]
            )
            return [make_search_result(document_id=query)]

    cases = [
        EvalCase(
            query=f"doc-{index}",
            namespace="rt",
            collections=("docs",),
            expected_ids=(f"doc-{index}",),
        )
        for index in range(4)
    ]

    results = asyncio.run(run_eval(cast(Engine, _DelayedCore()), cases, max_concurrency=3))

    assert [result.case.query for result in results] == [
        case.query for case in cases
    ]
    assert [result.retrieved_ids for result in results] == [
        (case.query,) for case in cases
    ]


def test_run_eval_concurrent_path_respects_max_concurrency() -> None:
    class _CountingCore:
        in_flight = 0
        high_water = 0

        async def search(self, **kwargs: object) -> list[object]:
            query = cast(str, kwargs["query"])
            self.in_flight += 1
            self.high_water = max(self.high_water, self.in_flight)
            try:
                await asyncio.sleep(0.01)
                return [make_search_result(document_id=query)]
            finally:
                self.in_flight -= 1

    cases = [
        EvalCase(
            query=f"doc-{index}",
            namespace="rt",
            collections=("docs",),
            expected_ids=(f"doc-{index}",),
        )
        for index in range(6)
    ]
    core = _CountingCore()

    results = asyncio.run(run_eval(cast(Engine, core), cases, max_concurrency=2))

    assert core.high_water == 2
    assert [result.error_type for result in results] == [None] * len(cases)


def test_run_eval_concurrent_path_isolates_case_failures() -> None:
    class _MixedCore:
        async def search(self, **kwargs: object) -> list[object]:
            query = cast(str, kwargs["query"])
            await asyncio.sleep(0.001)
            if query == "bad":
                raise RuntimeError("provider detail should not leak")
            return [make_search_result(document_id=query)]

    cases = [
        EvalCase(
            query="good-1",
            namespace="rt",
            collections=("docs",),
            expected_ids=("good-1",),
        ),
        EvalCase(
            query="bad",
            namespace="rt",
            collections=("docs",),
            expected_ids=("bad",),
        ),
        EvalCase(
            query="good-2",
            namespace="rt",
            collections=("docs",),
            expected_ids=("good-2",),
        ),
    ]

    results = asyncio.run(run_eval(cast(Engine, _MixedCore()), cases, max_concurrency=2))

    assert [result.case.query for result in results] == ["good-1", "bad", "good-2"]
    assert results[0].retrieved_ids == ("good-1",)
    assert results[1].retrieved_ids == ()
    assert results[1].error_type == "RuntimeError"
    assert "provider detail" not in repr(results[1])
    assert results[2].retrieved_ids == ("good-2",)


def test_eval_report_uses_wall_clock_throughput_for_concurrent_runs() -> None:
    results = [
        EvalResult(
            case=EvalCase(
                query=f"doc-{index}",
                namespace="rt",
                collections=("docs",),
                expected_ids=(f"doc-{index}",),
            ),
            retrieved_ids=(f"doc-{index}",),
            recall_at_5=1.0,
            recall_at_10=1.0,
            mrr=1.0,
            ndcg_at_10=1.0,
            latency_ms=100.0,
        )
        for index in range(2)
    ]

    report = eval_report(
        results,
        run={"max_concurrency": 2, "wall_clock_seconds": 0.1},
    )
    metrics = cast(dict[str, object], report["metrics"])

    assert metrics["throughput_qps"] == pytest.approx(20.0)
    assert metrics["serial_latency_qps"] == pytest.approx(10.0)


def test_run_eval_drives_core_search() -> None:
    async def go() -> list[EvalResult]:
        core = _make_keyword_core("rag_core_eval_run")
        try:
            await core.ensure_ready()
            for doc_id, body in (
                ("billing", "Billing happens monthly. Customers receive a billing email."),
                ("shipping", "Shipping rates depend on weight. International shipping uses customs."),
            ):
                await core.add_bytes(
                    file_bytes=f"# {doc_id}\n\n{body}".encode("utf-8"),
                    filename=f"{doc_id}.md",
                    mime_type="text/markdown",
                    namespace="rt",
                    collection="docs",
                    document_id=doc_id,
                    document_key=f"{doc_id}.md",
                )

            cases = [
                EvalCase(
                    query="when does billing run",
                    namespace="rt",
                    collections=("docs",),
                    expected_ids=("billing",),
                ),
                EvalCase(
                    query="international shipping customs",
                    namespace="rt",
                    collections=("docs",),
                    expected_ids=("shipping",),
                ),
            ]
            return await run_eval(core, cases)
        finally:
            await core.close()

    results = asyncio.run(go())

    assert len(results) == 2
    for result in results:
        assert result.recall_at_5 == 1.0
        assert result.recall_at_10 == 1.0
        assert result.mrr == 1.0
        assert result.ndcg_at_10 == 1.0
        assert result.latency_ms >= 0.0
        assert result.retrieved_ids[0] in result.case.expected_ids


def test_run_eval_passes_query_plan_to_core_search() -> None:
    async def go() -> RecordingVectorStore:
        store = RecordingVectorStore(
            search_results=[make_search_result(document_id="billing", collection="docs")]
        )
        core = Engine(
            make_test_config(
                qdrant_collection="rag_core_eval_query_plan",
                embedding_dimensions=4,
            ),
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=store,
        )
        case = EvalCase(
            query="billing policy",
            namespace="rt",
            collections=("docs",),
            expected_ids=("billing",),
        )
        plan = query_plan_preset("dense_only", limit=10)
        try:
            await run_eval(core, [case], query_plan=plan)
            return store
        finally:
            await core.close()

    store = asyncio.run(go())
    assert store.search_calls[0].query_plan == query_plan_preset("dense_only", limit=10)


def test_run_eval_matches_expected_document_key() -> None:
    class _DocumentKeyCore:
        async def search(self, **_: object) -> list[object]:
            return [
                make_search_result(
                    id="chunk-1",
                    document_id="internal-doc-id",
                    document_key="billing.md",
                )
            ]

    case = EvalCase(
        query="billing policy",
        namespace="rt",
        collections=("docs",),
        expected_ids=("billing.md",),
    )

    [result] = asyncio.run(run_eval(cast(Engine, _DocumentKeyCore()), [case]))

    assert result.retrieved_ids == ("billing.md",)
    assert result.recall_at_5 == 1.0
    assert result.mrr == 1.0


def test_run_eval_computes_context_quality_metrics() -> None:
    class _ContextCore:
        async def search(self, **_: object) -> list[object]:
            return [
                make_search_result(
                    id="chunk-1",
                    text="Invoices can be paid by ACH or card.",
                    document_id="private-doc-id",
                    document_key="private/billing.md",
                    title="Billing",
                    chunk_index=0,
                )
            ]

    case = EvalCase(
        query="invoice payment",
        namespace="rt",
        collections=("docs",),
        expected_ids=("private-doc-id",),
        expected_context_contains=("ACH", "card"),
        forbidden_context_contains=("# Metadata", "# Content"),
        forbidden_private_identifiers=("private/billing.md", "private-doc-id"),
        expected_citation_count_min=1,
        expected_source_count_min=1,
        max_context_chars=512,
    )

    [result] = asyncio.run(run_eval(cast(Engine, _ContextCore()), [case]))

    assert result.context_recall == 1.0
    assert result.context_contains_pass is True
    assert result.prompt_safety_pass is True
    assert result.forbidden_leak_count == 0
    assert result.citation_count == 1
    assert result.source_count == 1
    assert result.context_char_count <= 512


def test_run_eval_marks_context_forbidden_text_failures() -> None:
    class _LeakyCore:
        async def search(self, **_: object) -> list[object]:
            return [
                make_search_result(
                    id="chunk-1",
                    text="# Metadata\nsecret\n\n# Content\nBilling answer.",
                    document_id="billing",
                    title="Billing",
                )
            ]

    case = EvalCase(
        query="billing",
        namespace="rt",
        collections=("docs",),
        expected_ids=("billing",),
        expected_context_contains=("Billing answer",),
        forbidden_context_contains=("# Metadata", "# Content"),
    )

    [result] = asyncio.run(run_eval(cast(Engine, _LeakyCore()), [case]))

    assert result.context_recall == 1.0
    assert result.context_contains_pass is True
    assert result.prompt_safety_pass is False
    assert result.forbidden_leak_count == 2


def test_run_eval_records_search_failure_without_raw_exception_detail() -> None:
    class _FailingCore:
        async def search(self, **_: object) -> list[object]:
            raise RuntimeError("raw provider detail with api key sk-test-secret")

    case = EvalCase(
        query="billing policy",
        namespace="rt",
        collections=("docs",),
        expected_ids=("billing",),
    )

    results = asyncio.run(run_eval(cast(Engine, _FailingCore()), [case]))

    assert len(results) == 1
    result = results[0]
    assert result.retrieved_ids == ()
    assert result.recall_at_5 == 0.0
    assert result.error_type == "RuntimeError"
    assert "raw provider" not in repr(result)


@pytest.mark.eval_harness
def test_run_eval_can_measure_rerank_improvement() -> None:
    async def go() -> tuple[list[EvalResult], list[EvalResult], FakeReranker]:
        store = RecordingVectorStore(
            search_results=[
                make_search_result(
                    id="decoy-hit",
                    text="general account text",
                    score=0.9,
                    document_id="decoy",
                    collection="docs",
                ),
                make_search_result(
                    id="billing-hit",
                    text="billing policy and invoice schedule",
                    score=0.8,
                    document_id="billing",
                    collection="docs",
                ),
            ]
        )
        reranker = FakeReranker(
            results=[
                RerankResult(index=1, score=0.99, text="billing policy and invoice schedule"),
                RerankResult(index=0, score=0.10, text="general account text"),
            ]
        )
        core = Engine(
            make_test_config(
                qdrant_collection="rag_core_eval_rerank",
                embedding_dimensions=4,
            ),
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=store,
            reranker=reranker,
        )
        case = EvalCase(
            query="billing policy",
            namespace="rt",
            collections=("docs",),
            expected_ids=("billing",),
        )
        try:
            baseline = await run_eval(core, [case], rerank=False)
            reranked = await run_eval(
                core,
                [case],
                rerank=True,
                rerank_budget=RerankBudget(candidate_count=2, max_output=2),
            )
            return baseline, reranked, reranker
        finally:
            await core.close()

    baseline, reranked, reranker = asyncio.run(go())

    assert baseline[0].retrieved_ids == ("decoy", "billing")
    assert baseline[0].mrr == 0.5
    assert reranked[0].retrieved_ids == ("billing", "decoy")
    assert reranked[0].mrr == 1.0
    assert reranker.calls == [
        (
            "billing policy",
            ["general account text", "billing policy and invoice schedule"],
            2,
        )
    ]
