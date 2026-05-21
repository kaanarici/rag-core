from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import pytest

from rag_core.cli_eval import run_eval_command
from rag_core.cli_parser import _build_parser
from rag_core.core_models import RAGCoreConfig
from rag_core.evals import (
    EvalCase,
    EvalResult,
    eval_comparison_report,
    eval_profile_comparison_report,
    eval_report,
)
from rag_core.evals.reporting import format_eval_report
from rag_core.runtime_metadata import describe_runtime_metadata
from rag_core.search.planning import search_profile
from rag_core.search.types import SearchResult


def test_eval_report_preserves_run_metadata() -> None:
    report = eval_report(
        [_result()],
        run={
            "mode": "single",
            "vector_store": "qdrant",
            "search_profile": "balanced",
            "rerank": False,
        },
    )

    assert report["run"] == {
        "mode": "single",
        "vector_store": "qdrant",
        "search_profile": "balanced",
        "rerank": False,
    }
    assert format_eval_report(report)[0] == (
        "Run: mode=single vector_store=qdrant search_profile=balanced rerank=false"
    )


def test_eval_comparison_reports_preserve_branch_metadata() -> None:
    report = eval_comparison_report(
        [_result()],
        [_result()],
        baseline_run={"mode": "baseline", "rerank": False},
        reranked_run={
            "mode": "reranked",
            "rerank": True,
            "rerank_budget": {"candidate_count": 20},
        },
    )

    baseline = cast(dict[str, object], report["baseline"])
    reranked = cast(dict[str, object], report["reranked"])
    assert baseline["run"] == {"mode": "baseline", "rerank": False}
    assert reranked["run"] == {
        "mode": "reranked",
        "rerank": True,
        "rerank_budget": {"candidate_count": 20},
    }


def test_eval_profile_comparison_reports_preserve_profile_metadata() -> None:
    report = eval_profile_comparison_report(
        {
            "balanced": [_result()],
            "fast": [_result()],
        },
        profile_runs={
            "balanced": {"mode": "search_profile", "search_profile": "balanced"},
            "fast": {"mode": "search_profile", "search_profile": "fast"},
        },
    )

    profiles = cast(dict[str, dict[str, object]], report["profiles"])
    assert profiles["balanced"]["run"] == {
        "mode": "search_profile",
        "search_profile": "balanced",
    }
    assert profiles["fast"]["run"] == {
        "mode": "search_profile",
        "search_profile": "fast",
    }


def test_cli_eval_json_includes_run_shape(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "case_id": "billing",
                "query": "billing policy",
                "namespace": "acme",
                "corpus_ids": ["help"],
                "expected_chunk_ids": ["billing"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = _build_parser().parse_args(
        [
            "eval",
            "--cases",
            str(cases_path),
            "--search-profile",
            "balanced",
            "--rerank",
            "--rerank-candidates",
            "20",
            "--rerank-max-output",
            "5",
            "--rerank-timeout",
            "1.5",
            "--json",
        ]
    )

    exit_code = asyncio.run(
        run_eval_command(
            args,
            core_factory=lambda config: cast(Any, _FakeEvalCore(config)),
        )
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    runtime = describe_runtime_metadata()
    assert payload["run"] == {
        "embedding_model": "text-embedding-3-large",
        "embedding_provider": "openai",
        "mode": "single",
        "python_version": runtime["python_version"],
        "qdrant_collection": "rag_core_chunks",
        "rag_core_version": runtime["package_version"],
        "rerank": True,
        "rerank_budget": {
            "candidate_count": 20,
            "fallback_on_error": True,
            "max_output": 5,
            "timeout_seconds": 1.5,
        },
        "reranker_provider": "none",
        "search_profile": "balanced",
        "vector_store": "qdrant",
    }
    case_payload = cast(dict[str, object], payload["cases"][0])
    assert case_payload["case_label"] == "case-1"
    assert case_payload["corpus_count"] == 1
    assert "query" not in case_payload
    assert "namespace" not in case_payload
    assert "case_id" not in case_payload
    assert "corpus_ids" not in case_payload
    assert "expected_chunk_ids" not in case_payload
    assert "retrieved_ids" not in case_payload
    assert "billing policy" not in output
    assert "acme" not in output


def test_cli_eval_json_raw_explicitly_includes_case_payload(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "case_id": "billing",
                "query": "billing policy",
                "namespace": "acme",
                "corpus_ids": ["help"],
                "expected_chunk_ids": ["billing"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = _build_parser().parse_args(
        [
            "eval",
            "--cases",
            str(cases_path),
            "--json",
            "--json-raw",
        ]
    )

    exit_code = asyncio.run(
        run_eval_command(
            args,
            core_factory=lambda config: cast(Any, _FakeEvalCore(config)),
        )
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    case_payload = cast(dict[str, object], payload["cases"][0])
    assert case_payload["case_id"] == "billing"
    assert case_payload["query"] == "billing policy"
    assert case_payload["namespace"] == "acme"
    assert case_payload["corpus_ids"] == ["help"]
    assert case_payload["expected_chunk_ids"] == ["billing"]
    assert case_payload["retrieved_ids"] == ["billing"]


def test_cli_eval_json_raw_requires_json(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "case_id": "billing",
                "query": "billing policy",
                "namespace": "acme",
                "corpus_ids": ["help"],
                "expected_chunk_ids": ["billing"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = _build_parser().parse_args(
        [
            "eval",
            "--cases",
            str(cases_path),
            "--json-raw",
        ]
    )

    with pytest.raises(ValueError, match="--json-raw requires --json"):
        asyncio.run(
            run_eval_command(
                args,
                core_factory=lambda config: cast(Any, _FakeEvalCore(config)),
            )
        )


def test_cli_eval_normalizes_provider_bootstrap_errors(
    tmp_path: Path,
) -> None:
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "case_id": "billing",
                "query": "billing policy",
                "namespace": "acme",
                "corpus_ids": ["help"],
                "expected_chunk_ids": ["billing"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = _build_parser().parse_args(["eval", "--cases", str(cases_path)])

    def broken_core_factory(config: RAGCoreConfig) -> Any:
        raise RuntimeError("OpenAI API key missing")

    with pytest.raises(ValueError) as exc_info:
        asyncio.run(run_eval_command(args, core_factory=broken_core_factory))

    assert "provider setup failed before eval" in str(exc_info.value)
    assert "provider=openai" in str(exc_info.value)
    assert "OpenAI API key missing" not in str(exc_info.value)
    assert "rag-core doctor --json" in str(exc_info.value)


def test_cli_eval_compare_profiles_uses_deterministic_baseline_when_reordered(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "case_id": "billing",
                "query": "billing policy",
                "namespace": "acme",
                "corpus_ids": ["help"],
                "expected_chunk_ids": ["doc-1"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = _build_parser().parse_args(
        [
            "eval",
            "--cases",
            str(cases_path),
            "--compare-search-profiles",
            "fast",
            "balanced",
            "--json",
        ]
    )
    balanced_plan = search_profile("balanced", limit=10)
    fast_plan = search_profile("fast", limit=10)
    search_plans_seen: list[object] = []

    class _ProfileCore(_FakeEvalCore):
        async def search(self, **kwargs: object) -> list[SearchResult]:
            search_plans_seen.append(kwargs.get("query_plan"))
            if kwargs.get("query_plan") == fast_plan:
                return [
                    SearchResult(
                        id="doc-1",
                        text="Billing policy",
                        score=1.0,
                        content_type="document",
                        source_type="file",
                        document_id="doc-1",
                        corpus_id="help",
                    )
                ]
            return [
                SearchResult(
                    id="doc-2",
                    text="Other policy",
                    score=1.0,
                    content_type="document",
                    source_type="file",
                    document_id="doc-2",
                    corpus_id="help",
                )
            ]

    exit_code = asyncio.run(
        run_eval_command(
            args,
            core_factory=lambda config: cast(Any, _ProfileCore(config)),
        )
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert search_plans_seen == [balanced_plan, fast_plan]
    assert payload["baseline_profile"] == "balanced"
    assert payload["metric_deltas"]["fast"]["mrr"] == 1.0


def test_cli_eval_compare_rerank_gate_thresholds_ignore_baseline_metrics(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "case_id": "billing",
                "query": "billing policy",
                "namespace": "acme",
                "corpus_ids": ["help"],
                "expected_chunk_ids": ["billing"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = _build_parser().parse_args(
        [
            "eval",
            "--cases",
            str(cases_path),
            "--compare-rerank",
            "--min-recall-at-5",
            "1.0",
            "--json",
        ]
    )

    class _BaselineLowQualityCore(_FakeEvalCore):
        async def search(self, **kwargs: object) -> list[SearchResult]:
            if kwargs.get("rerank") is False:
                return [
                    SearchResult(
                        id="wrong",
                        text="Wrong policy",
                        score=1.0,
                        content_type="document",
                        source_type="file",
                        document_id="wrong",
                        corpus_id="help",
                    )
                ]
            return await super().search(**kwargs)

    exit_code = asyncio.run(
        run_eval_command(
            args,
            core_factory=lambda config: cast(Any, _BaselineLowQualityCore(config)),
        )
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    gate = cast(dict[str, object], payload["quality_gate"])
    assert gate["passed"] is True
    assert cast(list[object], gate["failures"]) == []
    overall = cast(dict[str, object], payload["overall"])
    assert overall == {"passed": True, "exit_code": 0}


def test_cli_eval_compare_rerank_gate_keeps_baseline_operational_failures_separate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "case_id": "billing",
                "query": "billing policy",
                "namespace": "acme",
                "corpus_ids": ["help"],
                "expected_chunk_ids": ["billing"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = _build_parser().parse_args(
        [
            "eval",
            "--cases",
            str(cases_path),
            "--compare-rerank",
            "--min-recall-at-5",
            "1.0",
            "--json",
        ]
    )

    class _BaselineFailingCore(_FakeEvalCore):
        async def search(self, **kwargs: object) -> list[SearchResult]:
            if kwargs.get("rerank") is False:
                raise RuntimeError("baseline failed")
            return await super().search(**kwargs)

    exit_code = asyncio.run(
        run_eval_command(
            args,
            core_factory=lambda config: cast(Any, _BaselineFailingCore(config)),
        )
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    gate = cast(dict[str, object], payload["quality_gate"])
    assert gate["passed"] is True
    failures = cast(list[object], gate["failures"])
    assert failures == []
    overall = cast(dict[str, object], payload["overall"])
    assert overall == {"passed": False, "exit_code": 1}


def _case() -> EvalCase:
    return EvalCase(
        case_id="billing",
        query="billing policy",
        namespace="acme",
        corpus_ids=("help",),
        expected_chunk_ids=("billing",),
    )


def _result() -> EvalResult:
    return EvalResult(
        case=_case(),
        retrieved_ids=("billing",),
        recall_at_5=1.0,
        recall_at_10=1.0,
        mrr=1.0,
        ndcg_at_10=1.0,
        latency_ms=1.0,
    )


class _FakeEvalCore:
    def __init__(self, config: RAGCoreConfig) -> None:
        self.config = config

    async def ensure_ready(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def search(self, **_: object) -> list[SearchResult]:
        return [
            SearchResult(
                id="billing",
                text="Billing policy",
                score=1.0,
                content_type="document",
                source_type="file",
                document_id="billing",
                corpus_id="help",
            )
        ]
