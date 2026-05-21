from __future__ import annotations

import json
from pathlib import Path

import pytest

import rag_core.cli as cli
import rag_core.events as events
from rag_core.events import summarize_search_trace_payload_runs


def test_summarize_search_trace_payload_runs_splits_eval_searches() -> None:
    assert "summarize_search_trace_payload_runs" in events.__all__

    summaries = summarize_search_trace_payload_runs(
        [
            {
                "event_type": "search.started",
                "namespace": "private-tenant",
                "corpus_ids": ["help"],
                "query_length": 12,
                "limit": 10,
            },
            {
                "event_type": "search.completed",
                "namespace": "private-tenant",
                "result_count": 1,
                "used_rerank": False,
                "used_sidecar": False,
                "duration_ms": 2.5,
            },
            {
                "event_type": "search.started",
                "namespace": "private-tenant",
                "corpus_ids": ["help", "internal"],
                "query_length": 19,
                "limit": 5,
            },
            {
                "event_type": "stage.error",
                "stage": "rerank",
                "error_type": "TimeoutError",
                "message": "raw provider detail for private query",
            },
            {
                "event_type": "search.completed",
                "namespace": "private-tenant",
                "result_count": 2,
                "used_rerank": True,
                "used_sidecar": False,
                "duration_ms": 7.5,
            },
        ]
    )

    assert len(summaries) == 2
    first = summaries[0].to_payload()
    second = summaries[1].to_payload()
    assert first["query_length"] == 12
    assert first["corpus_count"] == 1
    assert first["duration_ms"] == 2.5
    assert second["query_length"] == 19
    assert second["corpus_count"] == 2
    assert second["error_stages"] == ["rerank"]
    assert "private-tenant" not in repr(first)
    assert "raw provider detail" not in repr(second)


def test_summarize_search_trace_payload_runs_preserves_failed_terminal_summary() -> None:
    [summary] = summarize_search_trace_payload_runs(
        [
            {"event_type": "search.started", "query_length": 12, "limit": 5},
            {"event_type": "stage.error", "stage": "search", "error_type": "RuntimeError"},
            {
                "event_type": "search.completed",
                "result_count": 0,
                "requested_rerank": True,
                "requested_sidecar": True,
                "attempted_rerank": True,
                "attempted_sidecar": True,
                "succeeded": False,
                "duration_ms": 3.5,
            },
        ]
    )

    payload = summary.to_payload()
    assert payload["completed"] is True
    assert payload["succeeded"] is False
    assert payload["duration_ms"] == 3.5
    assert payload["requested_rerank"] is True
    assert payload["attempted_rerank"] is True
    assert payload["error_stages"] == ["search"]
    assert payload["error_types"] == ["RuntimeError"]


def test_jsonl_sink_preserves_search_ids_for_interleaved_runs(tmp_path: Path) -> None:
    trace_path = tmp_path / "events.jsonl"
    sink = events.JsonlSink(trace_path)

    sink.emit(events.SearchStarted(search_id="search-a", query_length=12, limit=10))
    sink.emit(events.SearchStarted(search_id="search-b", query_length=19, limit=5))
    sink.emit(events.SearchCompleted(search_id="search-a", result_count=1, duration_ms=2.5))
    sink.emit(events.SearchCompleted(search_id="search-b", result_count=2, duration_ms=7.5))

    payloads = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [payload["search_id"] for payload in payloads] == [
        "search-a",
        "search-b",
        "search-a",
        "search-b",
    ]

    summaries = summarize_search_trace_payload_runs(payloads)

    assert [summary.search_id for summary in summaries] == ["search-a", "search-b"]
    assert [summary.result_count for summary in summaries] == [1, 2]
    assert [summary.completed for summary in summaries] == [True, True]


def test_jsonl_and_trace_summary_drop_unsafe_search_ids_consistently(tmp_path: Path) -> None:
    trace_path = tmp_path / "events.jsonl"
    sink = events.JsonlSink(trace_path)

    sink.emit(events.SearchStarted(search_id="tenant/a", query_length=12, limit=10))
    sink.emit(events.SearchCompleted(search_id="tenant/a", result_count=1, duration_ms=2.5))

    payloads = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert all("search_id" not in payload for payload in payloads)

    [summary] = summarize_search_trace_payload_runs(payloads)
    assert summary.search_id == ""


def test_trace_summary_json_reports_multiple_searches(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = tmp_path / "eval.jsonl"
    _write_jsonl(
        trace_path,
        [
            {"event_type": "search.started", "query_length": 12, "limit": 10},
            {
                "event_type": "search.completed",
                "result_count": 1,
                "duration_ms": 2.5,
            },
            {"event_type": "search.started", "query_length": 19, "limit": 5},
            {
                "event_type": "rerank.applied",
                "provider": "cohere",
                "model": "rerank-v3.5",
                "input_count": 6,
                "candidate_count": 4,
                "provider_result_count": 5,
                "accepted_count": 3,
                "dropped_count": 2,
                "rank_changed_count": 3,
                "rank_promoted_count": 1,
                "rank_demoted_count": 2,
                "max_rank_gain": 2,
                "max_rank_loss": 1,
                "provider_score_min": 0.12,
                "provider_score_max": 0.91,
                "search_score_min": 0.2,
                "search_score_max": 0.9,
                "result_count": 3,
                "top_k": 3,
                "fallback_reason": "TimeoutError",
                "truncation_reason": "candidate_count,max_output",
                "duration_ms": 1.5,
                "succeeded": False,
            },
            {
                "event_type": "sidecar.applied",
                "provider": "bm25",
                "input_count": 5,
                "provider_result_count": 7,
                "accepted_count": 4,
                "dropped_count": 3,
                "result_count": 4,
                "duration_ms": 0.75,
                "succeeded": True,
            },
            {
                "event_type": "search.completed",
                "result_count": 2,
                "duration_ms": 7.5,
            },
        ],
    )

    exit_code = cli.main(["trace-summary", str(trace_path), "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["search_count"] == 2
    assert payload["completed_count"] == 2
    assert payload["total_duration_ms"] == 10.0
    assert payload["rerank_applied_count"] == 0
    assert payload["rerank_failed_count"] == 1
    assert payload["rerank_provider_result_count"] == 5
    assert payload["rerank_accepted_count"] == 3
    assert payload["rerank_dropped_count"] == 2
    assert payload["rerank_rank_changed_count"] == 3
    assert payload["rerank_rank_promoted_count"] == 1
    assert payload["rerank_rank_demoted_count"] == 2
    assert payload["rerank_max_rank_gain"] == 2
    assert payload["rerank_max_rank_loss"] == 1
    assert payload["rerank_provider_score_min"] == 0.12
    assert payload["rerank_provider_score_max"] == 0.91
    assert payload["rerank_search_score_min"] == 0.2
    assert payload["rerank_search_score_max"] == 0.9
    assert payload["rerank_duration_ms"] == 1.5
    assert payload["sidecar_applied_count"] == 1
    assert payload["sidecar_failed_count"] == 0
    assert payload["sidecar_provider_result_count"] == 7
    assert payload["sidecar_accepted_count"] == 4
    assert payload["sidecar_dropped_count"] == 3
    assert payload["sidecar_duration_ms"] == 0.75
    assert [search["query_length"] for search in payload["searches"]] == [12, 19]


def test_trace_summary_json_does_not_count_empty_rerank_as_applied(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = tmp_path / "eval.jsonl"
    _write_jsonl(
        trace_path,
        [
            {"event_type": "search.started", "query_length": 12, "limit": 10},
            {
                "event_type": "search.completed",
                "result_count": 1,
                "duration_ms": 2.5,
            },
            {"event_type": "search.started", "query_length": 19, "limit": 5},
            {
                "event_type": "rerank.applied",
                "provider": "cohere",
                "model": "rerank-v3.5",
                "input_count": 2,
                "candidate_count": 2,
                "provider_result_count": 0,
                "accepted_count": 0,
                "dropped_count": 0,
                "result_count": 2,
                "top_k": 2,
                "duration_ms": 1.5,
                "succeeded": True,
            },
            {
                "event_type": "search.completed",
                "result_count": 2,
                "duration_ms": 7.5,
                "requested_rerank": True,
                "attempted_rerank": True,
                "applied_rerank": False,
                "used_rerank": False,
            },
        ],
    )

    exit_code = cli.main(["trace-summary", str(trace_path), "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rerank_applied_count"] == 0
    assert payload["rerank_failed_count"] == 0
    assert payload["searches"][1]["rerank_succeeded"] is True
    assert payload["searches"][1]["rerank_applied"] is False
    assert payload["searches"][1]["applied_rerank"] is False


def test_trace_summary_text_reports_multi_search_applied_diagnostics(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = tmp_path / "eval.jsonl"
    _write_jsonl(
        trace_path,
        [
            {"event_type": "search.started", "query_length": 12, "limit": 10},
            {
                "event_type": "search.completed",
                "result_count": 1,
                "duration_ms": 2.5,
            },
            {"event_type": "search.started", "query_length": 19, "limit": 5},
            {
                "event_type": "rerank.applied",
                "provider": "cohere",
                "model": "rerank-v3.5",
                "input_count": 6,
                "candidate_count": 4,
                "provider_result_count": 5,
                "accepted_count": 3,
                "dropped_count": 2,
                "rank_changed_count": 3,
                "rank_promoted_count": 1,
                "rank_demoted_count": 2,
                "max_rank_gain": 2,
                "max_rank_loss": 1,
                "provider_score_min": 0.12,
                "provider_score_max": 0.91,
                "search_score_min": 0.2,
                "search_score_max": 0.9,
                "result_count": 3,
                "top_k": 3,
                "fallback_reason": "",
                "truncation_reason": "candidate_count,max_output",
                "duration_ms": 1.5,
                "succeeded": True,
            },
            {
                "event_type": "sidecar.applied",
                "provider": "bm25",
                "input_count": 5,
                "provider_result_count": 7,
                "accepted_count": 4,
                "dropped_count": 3,
                "result_count": 4,
                "duration_ms": 0.75,
                "succeeded": True,
            },
            {
                "event_type": "search.completed",
                "result_count": 2,
                "duration_ms": 7.5,
            },
        ],
    )

    exit_code = cli.main(["trace-summary", str(trace_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Searches: count=2 completed=2 errors=0 total_duration_ms=10.0" in output
    assert "Rerank aggregate: applied=1 failed=0 provider_results=5" in output
    assert "accepted=3 dropped=2 rank_changed=3 promoted=1 demoted=2" in output
    assert "max_gain=2 max_loss=1 provider_score=0.12..0.91" in output
    assert "search_score=0.2..0.9 duration_ms=1.5" in output
    assert "Sidecar aggregate: applied=1 failed=0 provider_results=7" in output
    assert "accepted=4 dropped=3 duration_ms=0.75" in output
    assert "- search 2: completed=True limit=5 results=2" in output
    assert "Rerank: provider=cohere model=rerank-v3.5" in output
    assert "provider_results=5 accepted=3 dropped=2" in output
    assert "truncation=candidate_count,max_output" in output
    assert "Sidecar: provider=bm25 inputs=5" in output
    assert "provider_results=7 accepted=4 dropped=3" in output


def test_trace_summary_text_reports_failed_only_aggregate_blocks(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = tmp_path / "eval.jsonl"
    _write_jsonl(
        trace_path,
        [
            {"event_type": "search.started", "query_length": 12, "limit": 10},
            {
                "event_type": "search.completed",
                "result_count": 1,
                "duration_ms": 2.5,
            },
            {"event_type": "search.started", "query_length": 19, "limit": 5},
            {
                "event_type": "rerank.applied",
                "provider": "cohere",
                "model": "rerank-v3.5",
                "input_count": 6,
                "candidate_count": 4,
                "provider_result_count": 0,
                "accepted_count": 0,
                "dropped_count": 4,
                "duration_ms": 1.5,
                "succeeded": False,
            },
            {
                "event_type": "sidecar.applied",
                "provider": "bm25",
                "input_count": 5,
                "provider_result_count": 0,
                "accepted_count": 0,
                "dropped_count": 5,
                "duration_ms": 0.75,
                "succeeded": False,
            },
            {
                "event_type": "search.completed",
                "result_count": 2,
                "duration_ms": 7.5,
            },
        ],
    )

    exit_code = cli.main(["trace-summary", str(trace_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Rerank aggregate: applied=0 failed=1 provider_results=0" in output
    assert "Sidecar aggregate: applied=0 failed=1 provider_results=0" in output


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
