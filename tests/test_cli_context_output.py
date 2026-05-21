from __future__ import annotations

import json

import pytest

import rag_core.cli as cli


class FakeContextPack:
    def as_text(self) -> str:
        return "[C1] Billing guide\nInvoices can be paid by card."

    def to_payload(self) -> dict[str, object]:
        return {
            "query": "billing policy",
            "snippets": [],
            "citations": [],
            "source_previews": [],
            "citation_summary": "[C1] Billing guide",
            "dropped_count": 0,
            "max_snippets": 3,
            "max_chars": 1000,
            "max_tokens": None,
            "token_estimate": 11,
            "char_count": 44,
            "truncated": False,
        }


class RecordingCore:
    calls: list[dict[str, object]] = []
    search_called = False

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    async def ensure_ready(self) -> None:
        pass

    async def retrieve_context(self, **kwargs: object) -> FakeContextPack:
        self.calls.append(kwargs)
        return FakeContextPack()

    async def search(self, **_kwargs: object) -> list[object]:
        self.search_called = True
        return []

    async def close(self) -> None:
        pass


def test_retrieve_context_emits_canonical_context_payload(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    RecordingCore.calls = []
    RecordingCore.search_called = False
    monkeypatch.setattr(cli, "RAGCore", RecordingCore)

    exit_code = cli.main(
        [
            "retrieve-context",
            "billing policy",
            "--namespace",
            "acme",
            "--corpus-id",
            "help",
            "--limit",
            "3",
            "--max-context-chars",
            "1000",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["context_text"] == "[C1] Billing guide\nInvoices can be paid by card."
    assert payload["citation_summary"] == "[C1] Billing guide"
    assert "ok" not in payload
    assert RecordingCore.calls == [
        {
            "query": "billing policy",
            "namespace": "acme",
            "corpus_ids": ["help"],
            "limit": 3,
            "rerank": False,
            "query_plan": None,
            "metadata_filter": None,
            "max_chars": 1000,
            "max_tokens": None,
        }
    ]
    assert not RecordingCore.search_called


def test_retrieve_context_rejects_raw_json_mode_before_runtime_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "RAGCore", RecordingCore)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "retrieve-context",
                "billing policy",
                "--namespace",
                "acme",
                "--corpus-id",
                "help",
                "--json",
            ]
        )

    assert exc_info.value.code == 2


def test_retrieve_context_rejects_non_positive_budget_before_runtime_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "RAGCore", RecordingCore)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "retrieve-context",
                "billing policy",
                "--namespace",
                "acme",
                "--corpus-id",
                "help",
                "--max-context-chars",
                "0",
            ]
        )

    assert exc_info.value.code == 2
