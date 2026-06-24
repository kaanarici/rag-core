from __future__ import annotations

import json

import pytest

import rag_core.cli as cli


class FakeContext:
    def as_text(self) -> str:
        return "[C1] Billing guide\nInvoices can be paid by card."

    def as_prompt_text(self, *, context_order: object = "rank") -> str:
        if context_order == "extrema":
            return "[S1] Billing guide\nInvoices can be paid by card.\n\n[S2] Appendix"
        assert context_order == "rank"
        return "[S1] Billing guide\nInvoices can be paid by card."

    def to_payload(self) -> dict[str, object]:
        return {
            "context_text": "[C1] Billing guide\nInvoices can be paid by card.",
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

    async def context(self, **kwargs: object) -> FakeContext:
        self.calls.append(kwargs)
        return FakeContext()

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
    monkeypatch.setattr(cli, "Engine", RecordingCore)

    exit_code = cli.main(
        [
            "context",
            "billing policy",
            "--namespace",
            "acme",
            "--collection",
            "help",
            "--qdrant-location",
            ":memory:",
            "--limit",
            "3",
            "--max-context-chars",
            "1000",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["context_text"] == "[S1] Billing guide\nInvoices can be paid by card."
    assert payload["citation_summary"] == "[C1] Billing guide"
    assert "ok" not in payload
    assert RecordingCore.calls == [
        {
            "query": "billing policy",
            "namespace": "acme",
            "collections": ["help"],
            "limit": 3,
            "rerank": False,
            "query_plan": None,
            "content_types": None,
            "document_ids": None,
            "metadata_filter": None,
            "max_chars": 1000,
            "max_tokens": None,
        }
    ]
    assert not RecordingCore.search_called


def test_retrieve_context_default_and_explicit_rank_output_are_byte_identical(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "Engine", RecordingCore)
    argv = [
        "context",
        "billing policy",
        "--namespace",
        "acme",
        "--collection",
        "help",
        "--qdrant-location",
        ":memory:",
    ]

    RecordingCore.calls = []
    assert cli.main(argv) == 0
    default_output = capsys.readouterr().out

    RecordingCore.calls = []
    assert cli.main([*argv, "--context-order", "rank"]) == 0
    explicit_rank_output = capsys.readouterr().out

    assert default_output == explicit_rank_output
    assert "context_order" not in RecordingCore.calls[0]


def test_retrieve_context_default_limit_matches_context_pack_default(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    RecordingCore.calls = []
    monkeypatch.setattr(cli, "Engine", RecordingCore)

    exit_code = cli.main(
        [
            "context",
            "billing policy",
            "--namespace",
            "acme",
            "--collection",
            "help",
            "--qdrant-location",
            ":memory:",
        ]
    )

    assert exit_code == 0
    capsys.readouterr()
    assert RecordingCore.calls[0]["limit"] == 8


def test_retrieve_context_passes_scope_filters(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    RecordingCore.calls = []
    monkeypatch.setattr(cli, "Engine", RecordingCore)

    exit_code = cli.main(
        [
            "context",
            "billing policy",
            "--namespace",
            "acme",
            "--collection",
            "help",
            "--qdrant-location",
            ":memory:",
            "--content-type",
            "document",
            "--document-id",
            "doc-1",
        ]
    )

    assert exit_code == 0
    capsys.readouterr()
    assert RecordingCore.calls[0]["content_types"] == ["document"]
    assert RecordingCore.calls[0]["document_ids"] == ["doc-1"]


def test_search_context_json_flag_keeps_context_payload(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    RecordingCore.calls = []
    monkeypatch.setattr(cli, "Engine", RecordingCore)

    exit_code = cli.main(
        [
            "context",
            "billing policy",
            "--namespace",
            "acme",
            "--collection",
            "help",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["context_text"] == "[S1] Billing guide\nInvoices can be paid by card."
    assert RecordingCore.calls[0]["query"] == "billing policy"


def test_retrieve_context_rejects_non_positive_budget_before_runtime_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "Engine", RecordingCore)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "context",
                "billing policy",
                "--namespace",
                "acme",
                "--collection",
                "help",
                "--max-context-chars",
                "0",
            ]
        )

    assert exc_info.value.code == 2


def test_retrieve_context_context_order_extrema_threads_prompt_rendering(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    RecordingCore.calls = []
    monkeypatch.setattr(cli, "Engine", RecordingCore)

    exit_code = cli.main(
        [
            "context",
            "billing policy",
            "--namespace",
            "acme",
            "--collection",
            "help",
            "--qdrant-location",
            ":memory:",
            "--context-order",
            "extrema",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["context_text"].endswith("[S2] Appendix")
    assert payload["citation_summary"] == "[C1] Billing guide"
    assert "context_order" not in RecordingCore.calls[0]
