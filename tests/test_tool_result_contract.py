from __future__ import annotations

import json

from rag_core.contracts import (
    SupportsContextPackPromptPayload,
    search_user_documents_tool_result,
)


class _PackWithReservedFields:
    def as_prompt_text(self, *, context_order: object = "rank") -> str:
        assert context_order == "rank"
        return "canonical context"

    def to_prompt_payload(self) -> dict[str, object]:
        return {
            "ok": False,
            "context_text": "stale context",
            "query": "billing",
            "snippets": [],
            "citations": [],
            "source_previews": [],
            "citation_summary": "",
            "dropped_count": 0,
            "max_snippets": 5,
            "max_chars": 3000,
            "max_tokens": None,
            "token_estimate": 4,
            "char_count": 17,
            "truncated": False,
        }


class _PromptOnlyPack:
    def as_prompt_text(self, *, context_order: object = "rank") -> str:
        assert context_order == "rank"
        return "safe context"

    def to_prompt_payload(self) -> dict[str, object]:
        return {
            "query": "billing",
            "snippets": [],
            "citations": [],
            "source_previews": [],
            "citation_summary": "",
            "dropped_count": 0,
            "max_snippets": 5,
            "max_chars": 3000,
            "max_tokens": None,
            "token_estimate": 4,
            "char_count": 12,
            "truncated": False,
        }


def test_search_user_documents_tool_result_owns_reserved_output_fields() -> None:
    payload = search_user_documents_tool_result(_PackWithReservedFields())

    assert payload["ok"] is True
    assert payload["context_text"] == "canonical context"
    assert payload["query"] == "billing"


def test_prompt_payload_protocol_does_not_require_app_text_projection() -> None:
    pack: SupportsContextPackPromptPayload = _PromptOnlyPack()

    payload = search_user_documents_tool_result(pack)

    assert payload["ok"] is True
    assert payload["context_text"] == "safe context"


def test_search_user_documents_tool_result_default_and_explicit_rank_are_byte_identical() -> None:
    default_payload = search_user_documents_tool_result(_PromptOnlyPack())
    explicit_rank_payload = search_user_documents_tool_result(
        _PromptOnlyPack(),
        context_order="rank",
    )

    assert json.dumps(default_payload, sort_keys=True) == json.dumps(
        explicit_rank_payload,
        sort_keys=True,
    )


def test_search_user_documents_tool_result_threads_context_order_to_context_text() -> None:
    class _Pack(_PromptOnlyPack):
        def as_prompt_text(self, *, context_order: object = "rank") -> str:
            if context_order == "extrema":
                return "safe extrema context"
            return super().as_prompt_text(context_order=context_order)

    payload = search_user_documents_tool_result(_Pack(), context_order="extrema")

    assert payload["context_text"] == "safe extrema context"
    assert payload["snippets"] == []
