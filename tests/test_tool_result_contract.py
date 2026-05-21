from __future__ import annotations

from rag_core.contracts import search_user_documents_tool_result


class _PackWithReservedFields:
    def as_text(self) -> str:
        return "canonical context"

    def to_payload(self) -> dict[str, object]:
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


def test_search_user_documents_tool_result_owns_reserved_output_fields() -> None:
    payload = search_user_documents_tool_result(_PackWithReservedFields())

    assert payload["ok"] is True
    assert payload["context_text"] == "canonical context"
    assert payload["query"] == "billing"
