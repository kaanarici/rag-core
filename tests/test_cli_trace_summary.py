from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

import rag_core.cli as cli
from rag_core.events import JsonlSink, SearchCompleted, SearchPlanned, SearchStarted

AWS_ACCESS_KEY_LABEL = "AKIA1234567890ABCDEF"
PREFIXED_OPENAI_SECRET = "openai:sk-proj-abcdefghijklmnopqrstuvwxyz123456"
PREFIXED_ANTHROPIC_SECRET = "anthropic:sk-ant-api03-abcdefghijklmnopqrstuvwxyz123456"
PREFIXED_SLACK_XOXC_SECRET = (
    "slack:xoxc-123456789012-123456789012-abcdefghijklmnopqrstuvwxyz"
)


def test_trace_summary_json_reads_events_jsonl_without_private_content(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = tmp_path / "events.jsonl"
    _write_jsonl(
        trace_path,
        [
            {
                "event_type": "ingest.started",
                "filename": "/private/customer/report.pdf",
            },
            {
                "event_type": "search.started",
                "namespace": "tenant-secret",
                "corpus_ids": ["corpus-secret"],
                "query_length": len("private billing query"),
                "limit": 4,
            },
            {
                "event_type": "search.planned",
                "namespace": "tenant-secret",
                "corpus_ids": ["corpus-secret"],
                "limit": 4,
                "final_limit": 4,
                "channels": ["dense:dense:primary", "sparse:bm25:bm25"],
                "prefetch_limits": [8, 12],
                "fusion": "rrf",
                "plan_rerank": "provider",
                "rerank_fallback_on_error": False,
                "use_sidecar": True,
                "retrieve_stage": "private-stage-name-secret",
                "postprocesses": ["private-postprocess-secret"],
            },
            {
                "event_type": "search.stage.completed",
                "stage": "context_pack",
                "stage_name": "private-stage-name-secret",
                "candidate_count": 3,
                "result_count": 2,
                "dropped_count": 1,
                "truncated": True,
                "citation_count": 2,
                "duration_ms": 0.5,
            },
            {
                "event_type": "stage.error",
                "stage": "private rerank stage",
                "error_type": "RuntimeError",
                "message": "raw provider error for /private/customer/report.pdf",
            },
            {
                "event_type": "search.completed",
                "namespace": "tenant-secret",
                "result_count": 2,
                "used_rerank": True,
                "used_sidecar": True,
                "duration_ms": 10.0,
            },
        ],
    )

    exit_code = cli.main(["trace-summary", str(trace_path), "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["completed"] is True
    assert payload["channels"] == ["dense:dense:primary", "sparse:bm25:bm25"]
    assert payload["rerank_fallback_on_error"] is False
    assert payload["stage_count"] == 1
    assert payload["stages"][0]["stage"] == "context_pack"
    assert payload["stages"][0]["stage_name"] == "unknown"
    assert payload["stages"][0]["dropped_count"] == 1
    assert payload["retrieve_stage"] == "unknown"
    assert payload["postprocesses"] == ["unknown"]
    assert payload["error_stages"] == ["unknown"]
    assert payload["error_types"] == ["RuntimeError"]
    rendered = repr(payload)
    assert "private billing query" not in rendered
    assert "tenant-secret" not in rendered
    assert "corpus-secret" not in rendered
    assert "private rerank stage" not in rendered
    assert "private-stage-name-secret" not in rendered
    assert "private-postprocess-secret" not in rendered
    assert "/private/customer/report.pdf" not in rendered
    assert "raw provider error" not in rendered


def test_trace_summary_preserves_safe_corpus_count_from_jsonl_sink(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = tmp_path / "events.jsonl"
    sink = JsonlSink(trace_path)
    sink.emit(SearchStarted(corpus_ids=("private-a", "private-b"), query_length=7, limit=4))
    sink.emit(
        SearchPlanned(
            corpus_ids=("private-a", "private-b"),
            limit=4,
            final_limit=4,
        )
    )
    sink.emit(SearchCompleted(result_count=2))

    exit_code = cli.main(["trace-summary", str(trace_path), "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    stored = trace_path.read_text(encoding="utf-8")
    assert payload["corpus_count"] == 2
    assert '"corpus_count": 2' in stored
    assert "private-a" not in stored
    assert "private-b" not in stored


def test_trace_summary_text_output_is_compact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = tmp_path / "events.jsonl"
    _write_jsonl(
        trace_path,
        [
            {"event_type": "search.started", "query_length": 7, "limit": 2},
            {
                "event_type": "search.planned",
                "plan_rerank": "provider",
                "rerank_fallback_on_error": False,
                "use_sidecar": True,
            },
            {
                "event_type": "search.stage.completed",
                "stage": "retrieve",
                "stage_name": "HybridRetrieve",
                "candidate_count": 5,
                "result_count": 2,
                "duration_ms": 1.25,
            },
            {
                "event_type": "rerank.applied",
                "provider": "cohere",
                "model": "rerank-v3.5",
                "input_count": 5,
                "candidate_count": 4,
                "provider_result_count": 6,
                "accepted_count": 3,
                "dropped_count": 3,
                "rank_changed_count": 2,
                "rank_promoted_count": 1,
                "rank_demoted_count": 1,
                "max_rank_gain": 2,
                "max_rank_loss": 1,
                "provider_score_min": 0.1,
                "provider_score_max": 0.9,
                "search_score_min": 0.3,
                "search_score_max": 0.8,
                "result_count": 2,
                "top_k": 2,
                "fallback_reason": "TimeoutError",
                "truncation_reason": "candidate_count,max_output",
                "duration_ms": 2.5,
                "succeeded": False,
            },
            {
                "event_type": "sidecar.applied",
                "provider": "bm25",
                "input_count": 2,
                "provider_result_count": 4,
                "accepted_count": 3,
                "dropped_count": 1,
                "result_count": 3,
                "duration_ms": 0.75,
                "succeeded": True,
            },
        ],
    )

    exit_code = cli.main(["trace-summary", str(trace_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Completed: False" in output
    assert "Stages:" in output
    assert "rerank_fallback=False" in output
    assert "retrieve:HybridRetrieve" in output
    assert "Rerank: provider=cohere model=rerank-v3.5" in output
    assert "provider_results=6 accepted=3 dropped=3" in output
    assert "rank_changed=2 promoted=1 demoted=1" in output
    assert "max_gain=2 max_loss=1 provider_score=0.1..0.9" in output
    assert "search_score=0.3..0.8" in output
    assert "truncation=candidate_count,max_output" in output
    assert "Sidecar: provider=bm25 inputs=2" in output
    assert "provider_results=4 accepted=3 dropped=1" in output


def test_trace_summary_text_outputs_unknown_for_absent_rerank_scores(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = tmp_path / "events.jsonl"
    _write_jsonl(
        trace_path,
        [
            {
                "event_type": "search.started",
                "query_length": 7,
                "limit": 2,
            },
            {
                "event_type": "rerank.applied",
                "provider": "cohere",
                "model": "rerank-v3.5",
                "accepted_count": 2,
                "succeeded": True,
            },
        ],
    )

    exit_code = cli.main(["trace-summary", str(trace_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "provider_score=unknown" in output
    assert "search_score=unknown" in output
    assert "provider_score=0.0..0.0" not in output
    assert "search_score=0.0..0.0" not in output


def test_trace_summary_json_groups_interleaved_searches_by_search_id(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = tmp_path / "events.jsonl"
    _write_jsonl(
        trace_path,
        [
            {
                "event_type": "search.started",
                "search_id": "search-a",
                "query_length": 4,
                "limit": 2,
            },
            {
                "event_type": "search.started",
                "search_id": "search-b",
                "query_length": 8,
                "limit": 3,
            },
            {
                "event_type": "search.stage.completed",
                "search_id": "search-a",
                "stage": "retrieve",
                "stage_name": "HybridRetrieve",
                "result_count": 2,
            },
            {
                "event_type": "search.completed",
                "search_id": "search-b",
                "result_count": 3,
                "duration_ms": 2.0,
            },
            {
                "event_type": "search.completed",
                "search_id": "search-a",
                "result_count": 2,
                "duration_ms": 1.0,
            },
        ],
    )

    exit_code = cli.main(["trace-summary", str(trace_path), "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1
    assert payload["search_count"] == 2
    searches = cast(list[dict[str, object]], payload["searches"])
    assert [search["search_id"] for search in searches] == ["search-a", "search-b"]
    assert searches[0]["query_length"] == 4
    assert searches[0]["stage_count"] == 1
    assert searches[0]["result_count"] == 2
    assert searches[1]["query_length"] == 8
    assert searches[1]["stage_count"] == 0
    assert searches[1]["result_count"] == 3


def test_trace_summary_rejects_mixed_correlated_and_uncorrelated_search_events(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = tmp_path / "events.jsonl"
    _write_jsonl(
        trace_path,
        [
            {"event_type": "search.started", "search_id": "search-a"},
            {"event_type": "search.completed", "result_count": 1},
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["trace-summary", str(trace_path), "--json"])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "trace contains mixed correlated and uncorrelated search events" in captured.err


def test_trace_summary_json_reports_embedding_cache_aggregate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = tmp_path / "events.jsonl"
    _write_jsonl(
        trace_path,
        [
            {
                "event_type": "embed.requested",
                "provider": AWS_ACCESS_KEY_LABEL,
                "model": AWS_ACCESS_KEY_LABEL,
                "text_count": 2,
                "role": "dense",
            },
            {
                "event_type": "embed.requested",
                "provider": PREFIXED_OPENAI_SECRET,
                "model": PREFIXED_ANTHROPIC_SECRET,
                "text_count": 1,
                "role": "dense",
            },
            {
                "event_type": "embed.requested",
                "provider": PREFIXED_SLACK_XOXC_SECRET,
                "model": PREFIXED_SLACK_XOXC_SECRET,
                "text_count": 1,
                "role": "dense",
            },
            {
                "event_type": "embed.completed",
                "provider": "openai",
                "model": "text-embedding-3-small",
                "text_count": 2,
                "role": "dense",
                "duration_ms": 3.5,
                "cache_hits": 1,
                "cache_misses": 1,
                "cache_writes": 1,
                "cache_bypasses": 0,
            },
            {
                "event_type": "embed.completed",
                "provider": "fastembed",
                "model": "bm25",
                "text_count": 2,
                "role": "sparse",
                "duration_ms": 0.5,
            },
        ],
    )

    exit_code = cli.main(["trace-summary", str(trace_path), "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["search_count"] == 0
    assert payload["embedding"] == {
        "requested_event_count": 3,
        "completed_event_count": 2,
        "requested_text_count": 4,
        "completed_text_count": 4,
        "dense_completed_text_count": 2,
        "sparse_completed_text_count": 2,
        "cache_hits": 1,
        "cache_misses": 1,
        "cache_writes": 1,
        "cache_bypasses": 0,
        "duration_ms": 4.0,
        "providers": ["unknown", "openai", "fastembed"],
        "models": ["unknown", "text-embedding-3-small", "bm25"],
    }
    assert AWS_ACCESS_KEY_LABEL not in repr(payload)
    assert PREFIXED_OPENAI_SECRET not in repr(payload)
    assert PREFIXED_ANTHROPIC_SECRET not in repr(payload)
    assert PREFIXED_SLACK_XOXC_SECRET not in repr(payload)


def test_trace_summary_text_reports_embedding_cache_aggregate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = tmp_path / "events.jsonl"
    _write_jsonl(
        trace_path,
        [
            {"event_type": "embed.requested", "text_count": 3, "role": "dense"},
            {
                "event_type": "embed.completed",
                "provider": "openai",
                "model": "text-embedding-3-small",
                "text_count": 3,
                "role": "dense",
                "duration_ms": 4.25,
                "cache_hits": 2,
                "cache_misses": 1,
                "cache_writes": 1,
                "cache_bypasses": 0,
            },
        ],
    )

    exit_code = cli.main(["trace-summary", str(trace_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Searches: count=0 completed=0 errors=0 total_duration_ms=0" in output
    assert "Completed: False" not in output
    assert "Embeddings: requested_events=1 completed_events=1" in output
    assert "requested_texts=3 completed_texts=3 dense_texts=3 sparse_texts=0" in output
    assert "cache_hits=2 cache_misses=1 cache_writes=1 cache_bypasses=0" in output
    assert "duration_ms=4.25 providers=openai models=text-embedding-3-small" in output


def test_trace_summary_rejects_invalid_jsonl_without_echoing_line(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = tmp_path / "events.jsonl"
    trace_path.write_text(
        '{"event_type":"search.started"}\nnot json with private billing query\n',
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["trace-summary", str(trace_path), "--json"])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "trace file: line 2: invalid JSON" in captured.err
    assert str(trace_path) not in captured.err
    assert "private billing query" not in captured.err


def test_trace_summary_rejects_duplicate_json_object_keys(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = tmp_path / "events.jsonl"
    trace_path.write_text(
        '{"event_type":"search.started","limit":2,"limit":3}\n',
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["trace-summary", str(trace_path), "--json"])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "trace file: line 1: duplicate JSON key: limit" in captured.err
    assert str(trace_path) not in captured.err


def test_trace_summary_rejects_non_finite_json_constants_in_unused_fields(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = tmp_path / "events.jsonl"
    trace_path.write_text(
        '{"event_type":"search.started","unused_debug":NaN}\n',
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["trace-summary", str(trace_path), "--json"])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "trace file: line 1: invalid JSON constant" in captured.err
    assert str(trace_path) not in captured.err
    assert "NaN" not in captured.err


def test_trace_summary_rejects_non_object_jsonl_without_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = tmp_path / "events.jsonl"
    trace_path.write_text('["private billing query"]\n', encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["trace-summary", str(trace_path), "--json"])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "trace file: line 1: expected JSON object" in captured.err
    assert str(trace_path) not in captured.err
    assert "private billing query" not in captured.err


def test_trace_summary_rejects_malformed_embedding_role_without_echoing_value(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = tmp_path / "events.jsonl"
    _write_jsonl(
        trace_path,
        [
            {
                "event_type": "embed.completed",
                "role": "private billing query",
            }
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["trace-summary", str(trace_path), "--json"])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "trace file: line 1: trace field role must be dense or sparse" in captured.err
    assert str(trace_path) not in captured.err
    assert "private billing query" not in captured.err


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_trace_summary_rejects_non_finite_duration_tokens(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    token: str,
) -> None:
    trace_path = tmp_path / "events.jsonl"
    trace_path.write_text(
        (
            '{"event_type":"search.stage.completed",'
            '"stage":"retrieve",'
            f'"duration_ms":{token}}}\n'
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["trace-summary", str(trace_path), "--json"])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "trace file: line 1: invalid JSON constant" in captured.err
    assert str(trace_path) not in captured.err
    assert token not in captured.err


def test_trace_summary_rejects_unsafe_search_id_without_merging(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = tmp_path / "events.jsonl"
    _write_jsonl(
        trace_path,
        [
            {"event_type": "search.started", "search_id": "tenant/a"},
            {"event_type": "search.started", "search_id": "tenant/b"},
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["trace-summary", str(trace_path), "--json"])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "trace field search_id must be a safe search identifier" in captured.err


def test_trace_summary_rejects_malformed_event_without_echoing_value(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = tmp_path / "events.jsonl"
    _write_jsonl(
        trace_path,
        [
            {
                "event_type": "search.stage.completed",
                "stage": "private billing query",
                "stage_name": "HybridRetrieve",
            }
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["trace-summary", str(trace_path), "--json"])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert (
        "trace file: line 1: trace field stage must be a supported search stage"
        in captured.err
    )
    assert str(trace_path) not in captured.err
    assert "private billing query" not in captured.err


def test_trace_summary_rejects_unreadable_trace_file_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_dir = tmp_path / "events-dir"
    trace_dir.mkdir()

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["trace-summary", str(trace_dir), "--json"])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "trace file: unable to read trace file" in captured.err
    assert str(trace_dir) not in captured.err
    assert "Traceback" not in captured.err


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
