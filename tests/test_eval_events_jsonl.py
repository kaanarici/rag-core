from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import rag_core.cli as cli
from rag_core.cli_parser import _build_parser
from rag_core.core_models import RAGCoreConfig
from rag_core.events import EventSink, SearchCompleted, SearchStarted
from rag_core.search.types import SearchResult


def test_eval_subparser_accepts_events_jsonl() -> None:
    args = _build_parser().parse_args(
        [
            "eval",
            "--cases",
            "cases.jsonl",
            "--qdrant-url",
            "http://localhost:6333",
            "--events-jsonl",
            "/tmp/eval-events.jsonl",
        ]
    )

    assert args.command == "eval"
    assert args.events_jsonl == "/tmp/eval-events.jsonl"


def test_eval_command_writes_events_jsonl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cases_path = tmp_path / "cases.jsonl"
    events_path = tmp_path / "traces" / "eval.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "case_id": "billing",
                "query": "private billing question",
                "namespace": "acme",
                "corpus_ids": ["help"],
                "expected_chunk_ids": ["billing"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "RAGCore", _FakeEvalCore)

    exit_code = cli.main(
        [
            "eval",
            "--cases",
            str(cases_path),
            "--events-jsonl",
            str(events_path),
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["case_count"] == 1
    assert payload["cases"][0]["case_ordinal"] == 1
    rows = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["event_type"] for row in rows] == [
        "search.started",
        "search.completed",
    ]
    assert rows[0]["query_length"] == len("private billing question")
    assert "private billing question" not in events_path.read_text(encoding="utf-8")


def test_eval_command_fails_when_events_jsonl_sink_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cases_path = tmp_path / "cases.jsonl"
    events_path = tmp_path / "eval.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "case_id": "billing",
                "query": "billing question",
                "namespace": "acme",
                "corpus_ids": ["help"],
                "expected_chunk_ids": ["billing"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fail_payload(_: object) -> dict[str, object]:
        raise RuntimeError("sink failed")

    monkeypatch.setattr(cli, "RAGCore", _FakeEvalCore)
    monkeypatch.setattr("rag_core.events.sinks.event_to_jsonl_dict", fail_payload)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "eval",
                "--cases",
                str(cases_path),
                "--events-jsonl",
                str(events_path),
                "--json",
            ]
        )

    assert exc_info.value.code == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "events JSONL sink failed to write" in output.err


class _FakeEvalCore:
    def __init__(
        self,
        config: RAGCoreConfig,
        *,
        event_sink: EventSink | None = None,
        **_: Any,
    ) -> None:
        self.config = config
        self.event_sink = event_sink

    async def ensure_ready(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def search(
        self,
        *,
        query: str,
        namespace: str,
        corpus_ids: list[str],
        limit: int = 10,
        rerank: bool = False,
        **_: Any,
    ) -> list[SearchResult]:
        if self.event_sink is not None:
            self.event_sink.emit(
                SearchStarted(
                    namespace=namespace,
                    corpus_ids=tuple(corpus_ids),
                    query_length=len(query),
                    limit=limit,
                )
            )
            self.event_sink.emit(
                SearchCompleted(
                    namespace=namespace,
                    result_count=1,
                    used_rerank=rerank,
                )
            )
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
