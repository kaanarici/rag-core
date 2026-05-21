"""Unit tests for EvalCase/EvalResult and the runner shape."""

from __future__ import annotations

import asyncio
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import cast

import pytest

from rag_core import RAGCore
from rag_core.evals import EvalCase, EvalResult, load_cases, run_eval
from rag_core.search.planning import query_plan_preset
from rag_core.search.providers.memory_store import InMemoryVectorStore
from rag_core.search.types import RerankBudget, RerankResult
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
    case = EvalCase(query="q", namespace="ns", corpus_ids=("c",), expected_chunk_ids=("a",))
    with pytest.raises((FrozenInstanceError, AttributeError)):
        case.query = "other"  # type: ignore[misc]


def test_eval_result_is_frozen() -> None:
    case = EvalCase(query="q", namespace="ns", corpus_ids=("c",), expected_chunk_ids=("a",))
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
            "corpus_ids": ["c"],
            "expected_chunk_ids": ["x", "y"],
        },
        {
            "query": "q2",
            "namespace": "ns",
            "corpus_ids": ["c"],
            "expected_chunk_ids": ["z", "w"],
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
        corpus_ids=("c",),
        expected_chunk_ids=("x", "y"),
        expected_grades=None,
    )
    assert cases[1].expected_chunk_ids == ("z", "w")
    assert cases[1].expected_grades == {"z": 3, "w": 1}


def test_load_cases_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "\n"
        + json.dumps(
            {
                "query": "q",
                "namespace": "ns",
                "corpus_ids": ["c"],
                "expected_chunk_ids": ["x"],
            }
        )
        + "\n\n",
        encoding="utf-8",
    )

    assert len(load_cases(path)) == 1


def _make_keyword_core(collection: str) -> RAGCore:
    return RAGCore(
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


def test_run_eval_drives_core_search() -> None:
    async def go() -> list[EvalResult]:
        core = _make_keyword_core("rag_core_eval_run")
        try:
            await core.ensure_ready()
            for doc_id, body in (
                ("billing", "Billing happens monthly. Customers receive a billing email."),
                ("shipping", "Shipping rates depend on weight. International shipping uses customs."),
            ):
                await core.ingest_bytes(
                    file_bytes=f"# {doc_id}\n\n{body}".encode("utf-8"),
                    filename=f"{doc_id}.md",
                    mime_type="text/markdown",
                    namespace="rt",
                    corpus_id="docs",
                    document_id=doc_id,
                    document_key=f"{doc_id}.md",
                )

            cases = [
                EvalCase(
                    query="when does billing run",
                    namespace="rt",
                    corpus_ids=("docs",),
                    expected_chunk_ids=("billing",),
                ),
                EvalCase(
                    query="international shipping customs",
                    namespace="rt",
                    corpus_ids=("docs",),
                    expected_chunk_ids=("shipping",),
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
        assert result.retrieved_ids[0] in result.case.expected_chunk_ids


def test_run_eval_passes_query_plan_to_core_search() -> None:
    async def go() -> RecordingVectorStore:
        store = RecordingVectorStore(
            search_results=[make_search_result(document_id="billing", corpus_id="docs")]
        )
        core = RAGCore(
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
            corpus_ids=("docs",),
            expected_chunk_ids=("billing",),
        )
        plan = query_plan_preset("dense_only", limit=10)
        try:
            await run_eval(core, [case], query_plan=plan)
            return store
        finally:
            await core.close()

    store = asyncio.run(go())
    assert store.search_calls[0].query_plan == query_plan_preset("dense_only", limit=10)


def test_run_eval_records_search_failure_without_raw_exception_detail() -> None:
    class _FailingCore:
        async def search(self, **_: object) -> list[object]:
            raise RuntimeError("raw provider detail with api key sk-test-secret")

    case = EvalCase(
        query="billing policy",
        namespace="rt",
        corpus_ids=("docs",),
        expected_chunk_ids=("billing",),
    )

    results = asyncio.run(run_eval(cast(RAGCore, _FailingCore()), [case]))

    assert len(results) == 1
    result = results[0]
    assert result.retrieved_ids == ()
    assert result.recall_at_5 == 0.0
    assert result.error_type == "RuntimeError"
    assert "raw provider" not in repr(result)


@pytest.mark.eval
def test_run_eval_can_measure_rerank_improvement() -> None:
    async def go() -> tuple[list[EvalResult], list[EvalResult], FakeReranker]:
        store = RecordingVectorStore(
            search_results=[
                make_search_result(
                    id="decoy-hit",
                    text="general account text",
                    score=0.9,
                    document_id="decoy",
                    corpus_id="docs",
                ),
                make_search_result(
                    id="billing-hit",
                    text="billing policy and invoice schedule",
                    score=0.8,
                    document_id="billing",
                    corpus_id="docs",
                ),
            ]
        )
        reranker = FakeReranker(
            results=[
                RerankResult(index=1, score=0.99, text="billing policy and invoice schedule"),
                RerankResult(index=0, score=0.10, text="general account text"),
            ]
        )
        core = RAGCore(
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
            corpus_ids=("docs",),
            expected_chunk_ids=("billing",),
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
