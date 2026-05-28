from __future__ import annotations

from rag_core.contracts import (
    SupportsContextPackPromptPayload,
    search_user_documents_tool_result,
)


class _PackWithReservedFields:
    def as_prompt_text(self) -> str:
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
    def as_prompt_text(self) -> str:
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
