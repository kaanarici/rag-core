import json
from pathlib import Path
from typing import cast

import pytest

from rag_core.contracts import (
    SEARCH_USER_DOCUMENTS_INPUT_SCHEMA,
    SEARCH_USER_DOCUMENTS_OUTPUT_SCHEMA,
    SEARCH_USER_DOCUMENTS_TOOL_NAME,
    search_user_documents_tool_contract,
    search_user_documents_tool_result,
)
from rag_core.search.context_pack import (
    ContextSnippet,
    ContextPack,
    SourceLocator,
    SourceReference,
)


# Stable shape of the tool input/output schemas — touched only when the
# advertised tool contract intentionally changes.
_INPUT_FIELDS = {
    "query",
    "limit",
    "document_ids",
    "rerank",
    "use_lexical_search",
    "max_chars",
    "max_tokens",
}
_OUTPUT_FIELDS = {
    "ok",
    "query",
    "context_text",
    "snippets",
    "citations",
    "source_previews",
    "citation_summary",
    "dropped_count",
    "max_snippets",
    "max_chars",
    "max_tokens",
    "token_estimate",
    "char_count",
    "truncated",
}


class _PromptPackWithPayload:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def as_prompt_text(self) -> str:
        return "[S1] billing context"

    def to_prompt_payload(self) -> dict[str, object]:
        return self._payload


def _payload_with_snippet(**snippet_overrides: object) -> dict[str, object]:
    snippet: dict[str, object] = {
        "citation_id": "doc-1",
        "rank": 1,
        "text": "billing context",
        "score": 0.9,
        "source": {"citation_id": "doc-1"},
        "locator": {
            "chunk_index": 0,
            "section_path": None,
            "page_number": None,
            "page_index": None,
            "slide_number": None,
            "sheet_name": None,
            "row_range": None,
            "line_start": None,
            "line_end": None,
            "bbox": None,
            "figure_id": None,
            "figure_caption": None,
            "figure_thumbnail_url": None,
        },
        "token_estimate": 3,
        "char_count": 15,
        "truncated": False,
    }
    snippet.update(snippet_overrides)
    return {
        "query": "billing",
        "snippets": [snippet],
        "citations": [{"citation_id": "doc-1"}],
        "source_previews": [],
        "citation_summary": "",
        "dropped_count": 0,
        "max_snippets": 5,
        "max_chars": 3000,
        "max_tokens": None,
        "token_estimate": 3,
        "char_count": 15,
        "truncated": False,
    }


def test_search_user_documents_contract_is_json_serializable() -> None:
    payload = search_user_documents_tool_contract()
    decoded = json.loads(json.dumps(payload, sort_keys=True))

    assert decoded["tool_name"] == SEARCH_USER_DOCUMENTS_TOOL_NAME
    assert decoded["input_schema"]["title"] == "search_user_documents.input"
    assert decoded["output_schema"]["title"] == "search_user_documents.output"


def test_search_user_documents_contract_core_shape_is_stable() -> None:
    assert SEARCH_USER_DOCUMENTS_INPUT_SCHEMA["required"] == ["query"]
    input_props = SEARCH_USER_DOCUMENTS_INPUT_SCHEMA["properties"]
    assert isinstance(input_props, dict)
    assert set(input_props) == _INPUT_FIELDS
    assert input_props["limit"]["description"] == (
        "Maximum number of context snippets to return."
    )
    assert input_props["document_ids"]["description"] == (
        "Optional narrowing filter inside the app-bound document scope."
    )

    output_props = SEARCH_USER_DOCUMENTS_OUTPUT_SCHEMA["properties"]
    assert isinstance(output_props, dict)
    assert set(output_props) == _OUTPUT_FIELDS
    assert set(SEARCH_USER_DOCUMENTS_OUTPUT_SCHEMA["required"]) == _OUTPUT_FIELDS


def test_search_user_documents_output_schema_names_nested_payload_shapes() -> None:
    definitions = SEARCH_USER_DOCUMENTS_OUTPUT_SCHEMA["definitions"]
    assert isinstance(definitions, dict)

    assert "source_reference" not in definitions
    assert "source_locator" not in definitions
    assert "source_preview" not in definitions

    source_reference = definitions["prompt_source_reference"]
    assert isinstance(source_reference, dict)
    assert source_reference["additionalProperties"] is False
    assert source_reference["required"] == ["citation_id"]
    assert "source_id" not in source_reference["properties"]
    assert "result_id" not in source_reference["properties"]
    assert "section_id" not in source_reference["properties"]
    assert "document_key" not in source_reference["properties"]
    assert "corpus_id" not in source_reference["properties"]
    assert "content_sha256" not in source_reference["properties"]
    assert "document_path" not in source_reference["properties"]

    source_locator = definitions["prompt_source_locator"]
    assert isinstance(source_locator, dict)
    assert "Prompt-safe source locator projection" in source_locator["description"]
    assert source_locator["additionalProperties"] is False
    assert "page_number" in source_locator["properties"]
    assert "line_start" in source_locator["properties"]
    assert "bbox" in source_locator["properties"]
    assert "source_hash" not in source_locator["properties"]

    snippet_items = SEARCH_USER_DOCUMENTS_OUTPUT_SCHEMA["properties"]["snippets"]["items"]
    citation_items = SEARCH_USER_DOCUMENTS_OUTPUT_SCHEMA["properties"]["citations"]["items"]
    preview_items = SEARCH_USER_DOCUMENTS_OUTPUT_SCHEMA["properties"]["source_previews"]["items"]
    assert snippet_items == {"$ref": "#/definitions/context_snippet"}
    assert citation_items == {"$ref": "#/definitions/prompt_source_reference"}
    assert preview_items == {"$ref": "#/definitions/prompt_source_preview"}

    snippet_schema = definitions["context_snippet"]
    assert isinstance(snippet_schema, dict)
    assert snippet_schema["properties"]["locator"] == {
        "$ref": "#/definitions/prompt_source_locator"
    }
    assert "retrieval_metadata" in snippet_schema["properties"]
    retrieval_metadata = snippet_schema["properties"]["retrieval_metadata"]
    assert isinstance(retrieval_metadata, dict)
    retrieval_properties = retrieval_metadata["properties"]
    assert isinstance(retrieval_properties, dict)
    assert "quality" in retrieval_properties
    quality = retrieval_properties["quality"]
    assert isinstance(quality, dict)
    assert quality["additionalProperties"] is False

    source_preview = definitions["prompt_source_preview"]
    assert isinstance(source_preview, dict)
    assert "document_key" not in source_preview["properties"]
    assert "corpus_id" not in source_preview["properties"]
    assert "source_hash" not in source_preview["properties"]


def test_tool_contract_returns_independent_copies() -> None:
    payload = search_user_documents_tool_contract()
    input_schema = payload["input_schema"]
    assert isinstance(input_schema, dict)
    input_schema["title"] = "mutated"

    fresh = search_user_documents_tool_contract()
    assert fresh["input_schema"]["title"] == "search_user_documents.input"


def test_search_user_documents_tool_result_wraps_context_pack_payload() -> None:
    class _Pack:
        def as_prompt_text(self) -> str:
            return "[S1] billing context"

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
                "token_estimate": 0,
                "char_count": 0,
                "truncated": False,
            }

    assert search_user_documents_tool_result(_Pack()) == {
        "ok": True,
        "context_text": "[S1] billing context",
        "query": "billing",
        "snippets": [],
        "citations": [],
        "source_previews": [],
        "citation_summary": "",
        "dropped_count": 0,
        "max_snippets": 5,
        "max_chars": 3000,
        "max_tokens": None,
        "token_estimate": 0,
        "char_count": 0,
        "truncated": False,
    }


def test_tool_contracts_do_not_keep_parallel_context_pack_projection_helpers() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "src" / "rag_core" / "contracts" / "tool_contracts.py").read_text(
        encoding="utf-8"
    )

    assert "_context_pack_tool_payload" not in source
    assert "_context_pack_tool_text" not in source


def test_search_user_documents_tool_result_uses_prompt_safe_context_payload() -> None:
    pack = ContextPack(
        query="billing",
        snippets=(
            ContextSnippet(
                citation_id="doc-1",
                rank=1,
                text="billing context",
                score=0.95,
                source=SourceReference(
                    source_id="doc-1",
                    result_id="hit-1",
                    document_id="internal-doc-id",
                    corpus_id="tenant-corpus",
                    document_key="private/billing.md",
                    title="Billing",
                    section_id="private-section-id",
                    section_title="Section title",
                    content_sha256="abc123",
                    source_type="document",
                    result_type="chunk",
                ),
                locator=SourceLocator(
                    chunk_index=0,
                    source_hash="abc123",
                    bbox=(1.0, 2.0, 3.0, 4.0),
                ),
                token_estimate=3,
                char_count=15,
                truncated=False,
            ),
        ),
        dropped_count=0,
        max_snippets=5,
        max_chars=3000,
        max_tokens=None,
        token_estimate=3,
    )

    result = search_user_documents_tool_result(pack)

    assert result["query"] == "billing"
    snippets = cast(list[object], result["snippets"])
    snippet = cast(dict[str, object], snippets[0])
    source = snippet["source"]
    assert isinstance(source, dict)
    assert "document_id" not in source
    assert "corpus_id" not in source
    assert "document_key" not in source
    assert "content_sha256" not in source
    assert "source_id" not in source
    assert "result_id" not in source
    assert "section_id" not in source
    assert source["citation_id"] == "S1"
    assert source["section_title"] == "Section title"
    locator = snippet["locator"]
    assert isinstance(locator, dict)
    assert locator["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert "source_hash" not in locator
    previews = cast(list[object], result["source_previews"])
    preview = cast(dict[str, object], previews[0])
    assert "document_id" not in preview
    assert "corpus_id" not in preview
    assert "document_key" not in preview
    assert "source_hash" not in preview


def test_search_user_documents_tool_result_does_not_use_private_ids_as_titles() -> None:
    pack = ContextPack(
        query="billing",
        snippets=(
            ContextSnippet(
                citation_id="doc-1",
                rank=1,
                text="billing context",
                score=0.95,
                source=SourceReference(
                    source_id="source-1",
                    result_id="hit-1",
                    document_id="internal-doc-id",
                    document_key="private/billing.md",
                    title=None,
                    source_type="document",
                    result_type="chunk",
                ),
                locator=SourceLocator(chunk_index=0),
                token_estimate=3,
                char_count=15,
                truncated=False,
            ),
        ),
        dropped_count=0,
        max_snippets=5,
        max_chars=3000,
        max_tokens=None,
        token_estimate=3,
    )

    result = search_user_documents_tool_result(pack)
    encoded = json.dumps(result, sort_keys=True)

    assert "[S1] document" in cast(str, result["context_text"])
    assert "source-1" not in encoded
    assert "hit-1" not in encoded
    assert "private/billing.md" not in encoded
    assert "internal-doc-id" not in encoded


def test_search_user_documents_tool_result_rejects_incomplete_payload() -> None:
    try:
        search_user_documents_tool_result(
            _PromptPackWithPayload(
                {
                    "query": "billing",
                    "snippets": [],
                    "citations": [],
                    "source_previews": [],
                    "citation_summary": "",
                }
            )
        )
    except ValueError as exc:
        assert "dropped_count" in str(exc)
    else:
        raise AssertionError("expected incomplete tool payload to be rejected")


def test_search_user_documents_tool_result_rejects_incomplete_nested_payload() -> None:
    try:
        search_user_documents_tool_result(
            _PromptPackWithPayload(
                {
                    "query": "billing",
                    "snippets": [],
                    "citations": [{}],
                    "source_previews": [],
                    "citation_summary": "",
                    "dropped_count": 0,
                    "max_snippets": 5,
                    "max_chars": 3000,
                    "max_tokens": None,
                    "token_estimate": 0,
                    "char_count": 0,
                    "truncated": False,
                }
            )
        )
    except ValueError as exc:
        assert "citation_id" in str(exc)
    else:
        raise AssertionError("expected incomplete nested tool payload to be rejected")


def test_search_user_documents_tool_result_rejects_wrong_container_type() -> None:
    try:
        search_user_documents_tool_result(
            _PromptPackWithPayload(
                {
                    "query": "billing",
                    "snippets": "not-a-list",
                    "citations": [],
                    "source_previews": [],
                    "citation_summary": "",
                    "dropped_count": 0,
                    "max_snippets": 5,
                    "max_chars": 3000,
                    "max_tokens": None,
                    "token_estimate": 0,
                    "char_count": 0,
                    "truncated": False,
                }
            )
        )
    except ValueError as exc:
        assert "snippets" in str(exc)
    else:
        raise AssertionError("expected wrong container type to be rejected")


def test_search_user_documents_tool_result_rejects_extra_payload_fields() -> None:
    try:
        search_user_documents_tool_result(
            _PromptPackWithPayload(
                {
                    "query": "billing",
                    "snippets": [],
                    "citations": [],
                    "source_previews": [],
                    "citation_summary": "",
                    "dropped_count": 0,
                    "max_snippets": 5,
                    "max_chars": 3000,
                    "max_tokens": None,
                    "token_estimate": 0,
                    "char_count": 0,
                    "truncated": False,
                    "debug_payload": {},
                }
            )
        )
    except ValueError as exc:
        assert "debug_payload" in str(exc)
    else:
        raise AssertionError("expected extra tool payload fields to be rejected")


def test_search_user_documents_tool_result_rejects_nested_extra_fields() -> None:
    try:
        search_user_documents_tool_result(
            _PromptPackWithPayload(
                {
                    "query": "billing",
                    "snippets": [],
                    "citations": [
                        {
                            "citation_id": "S1",
                            "document_path": "/private/docs/billing.md",
                        }
                    ],
                    "source_previews": [],
                    "citation_summary": "",
                    "dropped_count": 0,
                    "max_snippets": 5,
                    "max_chars": 3000,
                    "max_tokens": None,
                    "token_estimate": 0,
                    "char_count": 0,
                    "truncated": False,
                }
            )
        )
    except ValueError as exc:
        assert "document_path" in str(exc)
    else:
        raise AssertionError("expected nested extra tool payload fields to be rejected")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_search_user_documents_tool_result_rejects_non_finite_scores(value: float) -> None:
    with pytest.raises(ValueError, match="payload.snippets\\[0\\].score"):
        search_user_documents_tool_result(
            _PromptPackWithPayload(_payload_with_snippet(score=value))
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_search_user_documents_tool_result_rejects_non_finite_bbox_values(
    value: float,
) -> None:
    payload = _payload_with_snippet()
    snippets = payload["snippets"]
    assert isinstance(snippets, list)
    snippet = snippets[0]
    assert isinstance(snippet, dict)
    locator = snippet["locator"]
    assert isinstance(locator, dict)
    locator["bbox"] = [0.0, 1.0, value, 3.0]

    with pytest.raises(ValueError, match="payload.snippets\\[0\\].locator.bbox\\[2\\]"):
        search_user_documents_tool_result(_PromptPackWithPayload(payload))
