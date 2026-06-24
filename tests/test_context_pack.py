import asyncio
import json
import re

import pytest

from rag_core import Context, Engine
from rag_core.events import EventBuffer
from rag_core.retrieval_defaults import DEFAULT_CONTEXT_LIMIT
from rag_core.search.context_pack import (
    CONTEXT_EXPANSION_AFTER_METADATA_KEY,
    CONTEXT_EXPANSION_BEFORE_METADATA_KEY,
    EvidenceSpan,
    build_context_pack,
    context_pack_response_payload,
    evidence_span_from_result,
)
from rag_core.events.types import SearchStageCompleted
from rag_core.search.context_pack import (
    source_locator_from_result,
    source_preview_from_snippet,
    source_reference_from_result,
)
from rag_core.search.planning import query_plan_preset
from tests.support import (
    FakeEmbeddingProvider,
    FakeSparseEmbedder,
    RecordingVectorStore,
    make_search_result,
    make_test_config,
)


def _context_order_pack(count: int) -> Context:
    return build_context_pack(
        [
            make_search_result(
                id=f"hit-{index}",
                text=f"context row {index}",
                document_id=f"doc-{index}",
                title=f"Document {index}",
                chunk_index=index,
            )
            for index in range(1, count + 1)
        ],
        query="context order",
        max_snippets=count,
    )


def _prompt_ranks(text: str) -> list[int]:
    return [int(match) for match in re.findall(r"^\[S(\d+)\]", text, flags=re.MULTILINE)]


@pytest.mark.parametrize(
    ("count", "expected_ranks"),
    [
        (1, [1]),
        (2, [1, 2]),
        (3, [1, 3, 2]),
        (4, [1, 3, 4, 2]),
        (5, [1, 3, 5, 4, 2]),
        (6, [1, 3, 5, 6, 4, 2]),
    ],
)
def test_context_pack_as_prompt_text_extrema_orders_n_1_to_6(
    count: int,
    expected_ranks: list[int],
) -> None:
    pack = _context_order_pack(count)

    assert _prompt_ranks(pack.as_prompt_text(context_order="extrema")) == expected_ranks
    assert _prompt_ranks(pack.as_prompt_text()) == list(range(1, count + 1))
    assert pack.as_prompt_text(context_order="rank") == pack.as_prompt_text()


def test_context_pack_as_prompt_text_extrema_reorders_only_prompt_body() -> None:
    pack = _context_order_pack(5)

    prompt_text = pack.as_prompt_text(context_order="extrema")

    assert prompt_text.startswith("[S1]")
    assert prompt_text.rsplit("\n\n", 1)[-1].startswith("[S2]")
    for rank in range(1, 6):
        assert prompt_text.count(f"[S{rank}]") == 1
        assert f"[S{rank}]" in pack.prompt_citation_summary
    assert _prompt_ranks(pack.as_prompt_text()) == [1, 2, 3, 4, 5]
    assert _prompt_ranks(pack.prompt_citation_summary) == [1, 2, 3, 4, 5]


def test_context_pack_context_order_keeps_app_payload_rank_order_byte_identical() -> None:
    pack = _context_order_pack(5)

    rank_payload = context_pack_response_payload(pack, context_order="rank")
    extrema_payload = context_pack_response_payload(pack, context_order="extrema")
    app_rank_payload = {k: v for k, v in rank_payload.items() if k != "context_text"}
    app_extrema_payload = {k: v for k, v in extrema_payload.items() if k != "context_text"}

    assert [snippet.rank for snippet in pack.snippets] == [1, 2, 3, 4, 5]
    assert [preview.citation_id for preview in pack.source_previews] == [
        snippet.citation_id for snippet in pack.snippets
    ]
    assert app_rank_payload == app_extrema_payload
    assert json.dumps(app_rank_payload, sort_keys=True) == json.dumps(
        app_extrema_payload,
        sort_keys=True,
    )
    assert rank_payload["context_text"] != extrema_payload["context_text"]
    assert _prompt_ranks(str(extrema_payload["context_text"])) == [1, 3, 5, 4, 2]


def test_context_pack_preserves_rank_order_and_source_references() -> None:
    first = make_search_result(
        id="point-1",
        text="Billing happens on the first day of each month.",
        score=0.91,
        document_id="billing-doc",
        collection="help",
        document_key="billing.md",
        title="Billing",
        section_path="Help > Billing",
        chunk_index=2,
        content_sha256="abc123",
        metadata={"page_number": 4, "page_index": 3, "bbox": [1, 2, 3, 4]},
    )
    second = make_search_result(
        id="point-2",
        text="ACH and card payments are supported.",
        score=0.72,
        document_id="payments-doc",
        collection="help",
        document_key="payments.md",
        title="Payments",
        chunk_index=0,
    )

    pack = build_context_pack([first, second], query="how do invoices get paid?")

    assert [snippet.rank for snippet in pack.snippets] == [1, 2]
    assert [snippet.citation_id for snippet in pack.snippets] == [
        "space-1:help:billing-doc#chunk-2",
        "space-1:help:payments-doc#chunk-0",
    ]
    assert pack.citations[0].document_key == "billing.md"
    assert pack.snippets[0].locator.source_hash == "abc123"
    assert pack.snippets[0].locator.page_number == 4
    assert pack.snippets[0].locator.page_index == 3
    assert pack.snippets[0].locator.bbox == (1.0, 2.0, 3.0, 4.0)
    payload = pack.to_payload()
    snippets_payload = payload["snippets"]
    assert isinstance(snippets_payload, list)
    first_payload = snippets_payload[0]
    assert isinstance(first_payload, dict)
    first_locator = first_payload["locator"]
    assert isinstance(first_locator, dict)
    assert first_locator["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert (
        "[space-1:help:billing-doc#chunk-2] Billing Help > Billing, page 4, chunk 2"
        in pack.as_text()
    )
    assert pack.source_previews[0].source_hash == "abc123"
    assert pack.source_previews[0].as_text() == (
        "[space-1:help:billing-doc#chunk-2] Billing (Help > Billing, page 4, chunk 2)"
    )
    assert pack.citation_summary.startswith(
        "[space-1:help:billing-doc#chunk-2] Billing"
    )
    assert pack.char_count == len(pack.as_text())


def test_context_pack_prompt_text_does_not_emit_index_metadata_wrappers() -> None:
    result = make_search_result(
        id="point-1",
        text="Invoices can be paid by ACH or card.",
        document_id="private-doc-id",
        document_key="private/billing.md",
        title="Billing",
        chunk_index=0,
        metadata={"source": "fixture", "department": "finance"},
    )

    pack = build_context_pack([result], query="invoice payment")
    prompt_text = pack.as_prompt_text()

    assert "Invoices can be paid by ACH or card." in prompt_text
    assert "# Metadata" not in prompt_text
    assert "# Content" not in prompt_text
    assert "private/billing.md" not in prompt_text
    assert "private-doc-id" not in prompt_text
    assert result.metadata["department"] == "finance"
    assert pack.snippets[0].retrieval_metadata is None


def test_context_pack_default_snippet_limit_uses_shared_context_default() -> None:
    results = [
        make_search_result(
            id=f"hit-{index}",
            document_id=f"doc-{index}",
            document_key=f"doc-{index}.md",
            text=f"context {index}",
            score=1.0,
        )
        for index in range(DEFAULT_CONTEXT_LIMIT + 2)
    ]

    pack = build_context_pack(results, query="context")

    assert len(pack.snippets) == DEFAULT_CONTEXT_LIMIT
    assert pack.max_snippets == DEFAULT_CONTEXT_LIMIT
    assert pack.dropped_count == 2
    assert pack.truncated is True


def test_context_pack_exposes_slide_and_sheet_locators() -> None:
    slide = make_search_result(
        id="slide-1",
        text="Quarterly slide summary.",
        document_id="deck",
        title="Deck",
        section_path="Slide 2",
        chunk_index=0,
        metadata={"slide_number": 2},
    )
    sheet = make_search_result(
        id="sheet-1",
        text="Signals table.",
        document_id="workbook",
        title="Workbook",
        chunk_index=1,
        metadata={"sheet_name": "Signals", "row_range": "11-20"},
    )
    sheet_with_section = make_search_result(
        id="sheet-2",
        text="More signals.",
        document_id="workbook",
        title="Workbook",
        section_path="Sheet: Signals (Rows 11-20)",
        chunk_index=2,
        metadata={"sheet_name": "Signals", "row_range": "11-20"},
    )

    pack = build_context_pack([slide, sheet, sheet_with_section], query="signals")

    assert pack.snippets[0].locator.slide_number == 2
    assert (
        pack.snippets[0].header
        == "[space-1:corpus-1:deck#chunk-0] Deck Slide 2, chunk 0"
    )
    assert pack.snippets[1].locator.sheet_name == "Signals"
    assert pack.snippets[1].locator.row_range == "11-20"
    assert (
        pack.snippets[1].header
        == "[space-1:corpus-1:workbook#chunk-1] Workbook sheet Signals, rows 11-20, chunk 1"
    )
    assert (
        pack.snippets[2].header
        == "[space-1:corpus-1:workbook#chunk-2] Workbook Sheet: Signals (Rows 11-20), chunk 2"
    )
    snippets = pack.to_payload()["snippets"]
    assert isinstance(snippets, list)
    second = snippets[1]
    assert isinstance(second, dict)
    locator = second["locator"]
    assert isinstance(locator, dict)
    assert locator["sheet_name"] == "Signals"
    assert locator["page_number"] is None


def test_context_pack_exposes_code_line_locators() -> None:
    pack = build_context_pack(
        [
            make_search_result(
                id="code-hit",
                text="def answer(query: str) -> str:\n    return query",
                source_type="file",
                title="answer.py",
                chunk_index=0,
                metadata={"line_start": 10, "line_end": 11},
            )
        ],
        query="answer",
    )

    [snippet] = pack.snippets
    assert snippet.locator.line_start == 10
    assert snippet.locator.line_end == 11
    assert "lines 10-11" in snippet.header
    payload = pack.to_payload()
    snippets = payload["snippets"]
    assert isinstance(snippets, list)
    first = snippets[0]
    assert isinstance(first, dict)
    locator = first["locator"]
    assert isinstance(locator, dict)
    assert locator["line_start"] == 10
    assert locator["line_end"] == 11
    assert pack.source_previews[0].locator_label == "lines 10-11, chunk 0"


def test_context_pack_dedupes_page_label_when_section_path_already_names_page() -> None:
    hit = make_search_result(
        id="page-4",
        text="Page text.",
        document_id="doc",
        title="Document",
        section_path="Page 4",
        chunk_index=0,
        metadata={"page_number": 4},
    )

    pack = build_context_pack([hit], query="page")

    assert (
        pack.snippets[0].header
        == "[space-1:corpus-1:doc#chunk-0] Document Page 4, chunk 0"
    )


def test_context_pack_payload_keeps_missing_locator_keys_explicit() -> None:
    pack = build_context_pack(
        [make_search_result(id="hit-1", document_id="doc", chunk_index=0)],
        query="missing locators",
    )

    snippets = pack.to_payload()["snippets"]
    assert isinstance(snippets, list)
    first = snippets[0]
    assert isinstance(first, dict)
    locator = first["locator"]
    assert isinstance(locator, dict)
    assert locator["chunk_index"] == 0
    # Every optional locator slot is present as None so callers can rely on shape.
    optional_keys = {
        "section_path",
        "source_hash",
        "page_number",
        "page_index",
        "slide_number",
        "sheet_name",
        "row_range",
        "line_start",
        "line_end",
        "start_offset",
        "end_offset",
        "bbox",
        "figure_id",
        "figure_caption",
        "figure_thumbnail_url",
    }
    assert optional_keys.issubset(locator)
    assert all(locator[key] is None for key in optional_keys)


def test_context_pack_dedupes_repeated_source_chunks_and_fills_budget() -> None:
    first = make_search_result(
        id="point-1",
        document_id=None,
        document_key="guide.md",
        chunk_index=1,
        text="first guide chunk",
    )
    duplicate = make_search_result(
        id="point-2",
        document_id=None,
        document_key="guide.md",
        chunk_index=1,
        text="duplicate guide chunk",
    )
    second = make_search_result(
        id="point-3",
        document_id=None,
        document_key="guide.md",
        chunk_index=2,
        text="second guide chunk",
    )
    dropped_by_limit = make_search_result(
        id="point-4",
        document_id=None,
        document_key=None,
        chunk_index=None,
        text="fallback chunk",
    )

    pack = build_context_pack(
        [first, duplicate, second, dropped_by_limit],
        query="guide",
        max_snippets=2,
    )

    citation_ids = [snippet.citation_id for snippet in pack.snippets]
    assert citation_ids[0] == "space-1:corpus-1:guide.md#chunk-1"
    assert citation_ids[1] == "space-1:corpus-1:guide.md#chunk-2"
    assert [snippet.rank for snippet in pack.snippets] == [1, 2]
    assert "duplicate guide chunk" not in pack.as_text()
    assert pack.dropped_count == 2
    assert pack.truncated is True


def test_context_pack_source_ids_ignore_hidden_duplicate_hits() -> None:
    first = make_search_result(
        id="point-1",
        document_key="guide.md",
        document_id=None,
        chunk_index=1,
        text="first guide chunk",
    )
    duplicate = make_search_result(
        id="point-2",
        document_key="guide.md",
        document_id=None,
        chunk_index=1,
        text="duplicate guide chunk",
    )

    pack = build_context_pack([first, duplicate], query="guide")

    assert [snippet.citation_id for snippet in pack.snippets] == [
        "space-1:corpus-1:guide.md#chunk-1"
    ]
    assert "duplicate guide chunk" not in pack.as_text()


def test_context_pack_keeps_hits_without_chunk_identity_distinct() -> None:
    first = make_search_result(
        id="point-1", document_id="guide", chunk_index=None, text="first guide hit"
    )
    second = make_search_result(
        id="point-2", document_id="guide", chunk_index=None, text="second guide hit"
    )

    pack = build_context_pack([first, second], query="guide")

    citation_ids = [snippet.citation_id for snippet in pack.snippets]
    assert citation_ids[0].startswith("space-1:corpus-1:guide-")
    assert citation_ids[1].startswith("space-1:corpus-1:guide-")
    assert citation_ids[0] != citation_ids[1]
    assert pack.dropped_count == 0


def test_context_pack_keeps_figure_hits_distinct_from_text_chunks() -> None:
    text = make_search_result(
        id="point-1",
        document_id="guide",
        chunk_index=1,
        result_type="text",
        text="paragraph hit",
    )
    figure = make_search_result(
        id="point-2",
        document_id="guide",
        chunk_index=1,
        result_type="figure",
        figure_id="fig-1",
        text="figure hit",
    )

    pack = build_context_pack([text, figure], query="guide")

    assert [snippet.citation_id for snippet in pack.snippets] == [
        "space-1:corpus-1:guide#chunk-1",
        "space-1:corpus-1:guide#chunk-1#figure-fig-1",
    ]
    assert [snippet.locator.figure_id for snippet in pack.snippets] == [None, "fig-1"]
    assert pack.dropped_count == 0


def test_context_pack_keeps_document_key_hits_without_corpus_distinct() -> None:
    first = make_search_result(
        id="point-1",
        document_id=None,
        collection=None,
        document_key="guide.md",
        chunk_index=0,
        text="first guide hit",
    )
    second = make_search_result(
        id="point-2",
        document_id=None,
        collection=None,
        document_key="guide.md",
        chunk_index=0,
        text="second guide hit",
    )

    pack = build_context_pack([first, second], query="guide")

    citation_ids = [snippet.citation_id for snippet in pack.snippets]
    assert citation_ids[0].startswith("space-1:guide.md#chunk-0-")
    assert citation_ids[1].startswith("space-1:guide.md#chunk-0-")
    assert citation_ids[0] != citation_ids[1]
    assert "second guide hit" in pack.as_text()
    assert pack.dropped_count == 0


def test_context_pack_duplicate_document_key_citation_ids_are_rank_stable() -> None:
    first = make_search_result(
        id="point-1",
        document_id=None,
        collection=None,
        document_key="guide.md",
        chunk_index=0,
        text="first guide hit",
    )
    second = make_search_result(
        id="point-2",
        document_id=None,
        collection=None,
        document_key="guide.md",
        chunk_index=0,
        text="second guide hit",
    )

    first_order = build_context_pack([first, second], query="guide")
    reversed_order = build_context_pack([second, first], query="guide")

    first_ids = {snippet.text: snippet.citation_id for snippet in first_order.snippets}
    reversed_ids = {
        snippet.text: snippet.citation_id for snippet in reversed_order.snippets
    }
    assert first_ids == reversed_ids
    assert first_ids["first guide hit"].startswith("space-1:guide.md#chunk-0-")
    assert first_ids["second guide hit"].startswith("space-1:guide.md#chunk-0-")
    assert first_ids["first guide hit"] != first_ids["second guide hit"]


def test_context_pack_citation_ids_are_collection_scoped_when_available() -> None:
    first = make_search_result(
        id="point-1",
        collection="corp-a",
        document_id="guide",
        chunk_index=1,
        text="first corpus hit",
    )
    second = make_search_result(
        id="point-2",
        collection="corp-b",
        document_id="guide",
        chunk_index=1,
        text="second corpus hit",
    )

    pack = build_context_pack([first, second], query="guide")

    assert [snippet.citation_id for snippet in pack.snippets] == [
        "space-1:corp-a:guide#chunk-1",
        "space-1:corp-b:guide#chunk-1",
    ]


def test_context_pack_dedupes_missing_chunk_when_section_identity_matches() -> None:
    first = make_search_result(
        id="point-1",
        document_id="guide",
        chunk_index=None,
        section_path="Guide > Setup",
        text="setup hit",
    )
    duplicate = make_search_result(
        id="point-2",
        document_id="guide",
        chunk_index=None,
        section_path="Guide > Setup",
        text="duplicate setup hit",
    )
    second = make_search_result(
        id="point-3",
        document_id="guide",
        chunk_index=None,
        section_path="Guide > Billing",
        text="billing hit",
    )

    pack = build_context_pack([first, duplicate, second], query="guide")

    assert len(pack.snippets) == 2
    assert all(
        snippet.citation_id.startswith("space-1:corpus-1:guide#section-")
        for snippet in pack.snippets
    )
    assert pack.snippets[0].citation_id != pack.snippets[1].citation_id
    assert "duplicate setup hit" not in pack.as_text()
    assert pack.dropped_count == 1
    assert pack.truncated is False


def test_context_pack_keeps_section_citation_ids_stable_when_rank_order_changes() -> (
    None
):
    setup = make_search_result(
        id="point-1",
        document_id="guide",
        chunk_index=None,
        section_path="Guide > Setup",
        text="setup hit",
    )
    billing = make_search_result(
        id="point-2",
        document_id="guide",
        chunk_index=None,
        section_path="Guide > Billing",
        text="billing hit",
    )

    first_order = build_context_pack([setup, billing], query="guide")
    reversed_order = build_context_pack([billing, setup], query="guide")

    first_ids = {snippet.text: snippet.citation_id for snippet in first_order.snippets}
    reversed_ids = {
        snippet.text: snippet.citation_id for snippet in reversed_order.snippets
    }
    assert first_ids == reversed_ids


def test_context_pack_enforces_char_budget_and_reports_drops() -> None:
    pack = build_context_pack(
        [
            make_search_result(
                id="a", text="abcdefghij", document_id="doc-a", chunk_index=0
            ),
            make_search_result(
                id="b", text="klmnop", document_id="doc-b", chunk_index=0
            ),
            make_search_result(
                id="c", text="qrstuv", document_id="doc-c", chunk_index=0
            ),
        ],
        query="budget",
        max_snippets=3,
        max_chars=80,
    )

    assert len(pack.as_text()) <= 80
    assert pack.char_count == len(pack.as_text())
    assert pack.dropped_count >= 1
    assert pack.truncated is True


def test_context_pack_budgets_rendered_text_not_only_raw_snippets() -> None:
    pack = build_context_pack(
        [
            make_search_result(
                id="a", text="abcdefghij", document_id="doc-a", chunk_index=0
            )
        ],
        query="budget",
        max_chars=3,
    )

    assert pack.as_text() == ""
    assert pack.char_count == 0
    assert pack.dropped_count == 1
    assert pack.truncated is True


def test_context_pack_token_budget_translates_to_char_budget() -> None:
    pack = build_context_pack(
        [
            make_search_result(
                id="a", text="abcdefghij", document_id="doc-a", chunk_index=0
            )
        ],
        query="budget",
        max_tokens=10,
        chars_per_token=2,
    )

    assert len(pack.as_text()) <= 20
    assert pack.max_chars == 20
    assert pack.char_count == len(pack.as_text())


def test_context_pack_uses_expansion_text_without_moving_locator() -> None:
    hit = make_search_result(
        id="expanded",
        text="Matched amberglint chunk.",
        document_id="guide",
        title="Guide",
        chunk_index=2,
        start_char=100,
        end_char=124,
        metadata={
            CONTEXT_EXPANSION_BEFORE_METADATA_KEY: ["Previous policy sentence."],
            CONTEXT_EXPANSION_AFTER_METADATA_KEY: ["Next rollout sentence."],
        },
    )

    pack = build_context_pack([hit], query="amberglint")
    [snippet] = pack.snippets

    assert snippet.text == (
        "Previous policy sentence.\n\n"
        "Matched amberglint chunk.\n\n"
        "Next rollout sentence."
    )
    assert snippet.locator.start_offset == 100
    assert snippet.locator.end_offset == 124
    assert pack.source_previews[0].locator_label == "chunk 2"


def test_context_pack_trims_expansion_before_original_chunk() -> None:
    hit = make_search_result(
        id="expanded",
        text="Matched amberglint chunk.",
        document_id="guide",
        title="Guide",
        chunk_index=2,
        start_char=100,
        end_char=124,
        metadata={
            CONTEXT_EXPANSION_BEFORE_METADATA_KEY: ["Previous policy sentence."],
            CONTEXT_EXPANSION_AFTER_METADATA_KEY: ["Next rollout sentence."],
        },
    )
    header = "[space-1:corpus-1:guide#chunk-2] Guide chunk 2"

    pack = build_context_pack(
        [hit],
        query="amberglint",
        max_chars=len(header) + 1 + len(hit.text),
    )
    [snippet] = pack.snippets

    assert snippet.text == hit.text
    assert snippet.truncated is True
    assert snippet.locator.start_offset == 100
    assert snippet.locator.end_offset == 124


@pytest.mark.parametrize(
    ("kwarg", "value"),
    [
        ("max_snippets", 0),
        ("max_chars", 0),
        ("max_tokens", 0),
    ],
)
def test_context_pack_rejects_non_positive_budgets(kwarg: str, value: int) -> None:
    with pytest.raises(ValueError, match=kwarg):
        build_context_pack([], query="x", **{kwarg: value})


def test_source_helpers_ignore_invalid_locator_metadata() -> None:
    result = make_search_result(
        id="point-1",
        metadata={"page_number": True, "bbox": [1, "bad", 3, 4]},
    )

    assert (
        source_reference_from_result(result).source_id
        == "space-1:corpus-1:doc-1#chunk-0"
    )
    locator = source_locator_from_result(result)
    assert locator.page_number is None
    assert locator.bbox is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_source_locator_drops_non_finite_bbox_values(value: float) -> None:
    result = make_search_result(
        id="point-1",
        metadata={"bbox": [1, 2, value, 4]},
    )

    locator = source_locator_from_result(result)
    assert locator.bbox is None


def test_source_locator_uses_figure_bbox_when_plain_bbox_is_absent() -> None:
    result = make_search_result(
        id="figure-point",
        figure_id="fig-1",
        metadata={"figure_bbox": [10, 20, 30, 40]},
    )

    locator = source_locator_from_result(result)

    assert locator.figure_id == "fig-1"
    assert locator.bbox == (10.0, 20.0, 30.0, 40.0)


def test_source_locator_reads_figure_thumbnail_url_from_result_or_metadata() -> None:
    from_result = source_locator_from_result(
        make_search_result(
            id="figure-point-1",
            figure_thumbnail_url="https://cdn.example.com/result.png",
        )
    )
    from_metadata = source_locator_from_result(
        make_search_result(
            id="figure-point-2",
            metadata={"figure_thumbnail_url": "https://cdn.example.com/metadata.png"},
        )
    )

    assert from_result.figure_thumbnail_url == "https://cdn.example.com/result.png"
    assert from_metadata.figure_thumbnail_url == "https://cdn.example.com/metadata.png"


def test_context_pack_preserves_sanitized_rerank_provenance() -> None:
    hit = make_search_result(
        id="reranked",
        text="reranked text",
        score=0.91,
        metadata={
            "rerank": {
                "provider": "voyage",
                "model": "rerank-2.5-lite",
                "provider_score": 0.91,
                "search_score": 0.42,
                "original_rank": 4,
                "rerank_rank": 1,
                "rank_delta": 3,
                "ignored": {"raw": "shape"},
            }
        },
    )

    pack = build_context_pack([hit], query="rerank")
    payload = pack.to_payload()
    snippets = payload["snippets"]
    assert isinstance(snippets, list)
    first = snippets[0]
    assert isinstance(first, dict)
    assert first["retrieval_metadata"] == {
        "rerank": {
            "provider": "voyage",
            "model": "rerank-2.5-lite",
            "provider_score": 0.91,
            "search_score": 0.42,
            "original_rank": 4,
            "rerank_rank": 1,
            "rank_delta": 3,
        }
    }


def test_context_pack_preserves_parse_quality_provenance() -> None:
    hit = make_search_result(
        id="quality-hit",
        text="quality text",
        score=0.8,
        metadata={
            "quality_verdict": "poor",
            "quality_details": "low meaningful text ratio",
            "quality_char_count": 128,
            "quality_meaningful_ratio": 0.31,
            "quality_mojibake_ratio": 0.04,
            "quality_text_to_page_ratio": 64.0,
            "quality_page_count": 2,
            "quality_ignored": {"raw": "shape"},
        },
    )

    pack = build_context_pack([hit], query="quality")
    payload = pack.to_payload()
    snippets = payload["snippets"]
    assert isinstance(snippets, list)
    first = snippets[0]
    assert isinstance(first, dict)
    assert first["retrieval_metadata"] == {
        "quality": {
            "verdict": "poor",
            "details": "low meaningful text ratio",
            "char_count": 128,
            "page_count": 2,
            "meaningful_ratio": 0.31,
            "mojibake_ratio": 0.04,
            "text_to_page_ratio": 64.0,
        }
    }


def test_context_pack_sanitizes_non_finite_scores() -> None:
    pack = build_context_pack(
        [make_search_result(id="bad-score", score=float("nan"))],
        query="score",
    )

    assert pack.snippets[0].score == 0.0
    snippets = pack.to_payload()["snippets"]
    assert isinstance(snippets, list)
    first = snippets[0]
    assert isinstance(first, dict)
    assert first["score"] == 0.0


def test_context_pack_drops_malformed_bbox_instead_of_failing() -> None:
    pack = build_context_pack(
        [
            make_search_result(
                id="bad-bbox",
                metadata={"bbox": [1.0, float("nan"), 3.0, 4.0]},
            )
        ],
        query="bbox",
    )

    assert pack.snippets[0].locator.bbox is None


def test_rag_core_retrieve_context_builds_pack_and_forwards_scope() -> None:
    async def scenario() -> tuple[Context, RecordingVectorStore]:
        store = RecordingVectorStore(
            search_results=[
                make_search_result(
                    id="hit-1",
                    text="Invoice payment context",
                    document_id="billing-doc",
                    collection="help",
                    chunk_index=3,
                )
            ]
        )
        core = Engine(
            make_test_config(
                qdrant_collection="rag_core_context_pack", embedding_dimensions=4
            ),
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=store,
        )
        try:
            pack = await core.context(
                query="How do I pay?",
                namespace="acme",
                collections=["help"],
                limit=4,
                rerank=False,
                max_chars=100,
            )
        finally:
            await core.close()
        return pack, store

    pack, store = asyncio.run(scenario())
    assert pack.query == "How do I pay?"
    assert pack.snippets[0].citation_id == "space-1:help:billing-doc#chunk-3"
    assert store.search_calls[0].limit == 4
    assert store.search_calls[0].collections == ["help"]


def test_rag_core_retrieve_context_honors_explicit_query_plan_final_limit() -> None:
    async def scenario() -> tuple[Context, RecordingVectorStore]:
        store = RecordingVectorStore(
            search_results=[
                make_search_result(
                    id="hit-1", text="One", document_id="doc-1", chunk_index=0
                ),
                make_search_result(
                    id="hit-2", text="Two", document_id="doc-2", chunk_index=0
                ),
                make_search_result(
                    id="hit-3", text="Three", document_id="doc-3", chunk_index=0
                ),
            ]
        )
        core = Engine(
            make_test_config(
                qdrant_collection="rag_core_context_pack_query_plan_limit",
                embedding_dimensions=4,
            ),
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=store,
        )
        try:
            pack = await core.context(
                query="How do I pay?",
                namespace="acme",
                collections=["help"],
                limit=1,
                rerank=False,
                query_plan=query_plan_preset("dense_only", limit=3),
            )
        finally:
            await core.close()
        return pack, store

    pack, store = asyncio.run(scenario())
    assert len(pack.snippets) == 3
    assert store.search_calls[0].limit == 3


def test_context_pack_payload_includes_source_previews_and_citation_summary() -> None:
    pack = build_context_pack(
        [
            make_search_result(
                id="hit-1",
                text="Invoice payment context",
                document_id="billing-doc",
                document_key="billing.md",
                title="Billing",
                chunk_index=3,
                content_sha256="sha256:abc",
                metadata={"page_number": 7},
            )
        ],
        query="billing",
    )

    payload = pack.to_payload()
    previews = payload["source_previews"]
    assert isinstance(previews, list)
    [preview] = previews
    assert preview["citation_id"] == "space-1:corpus-1:billing-doc#chunk-3"
    assert preview["title"] == "Billing"
    assert preview["locator_label"] == "page 7, chunk 3"
    assert preview["document_key"] == "billing.md"
    assert preview["source_hash"] == "sha256:abc"
    assert preview["truncated"] is False
    assert payload["citation_summary"] == (
        "[space-1:corpus-1:billing-doc#chunk-3] Billing (page 7, chunk 3)"
    )
    assert source_preview_from_snippet(pack.snippets[0]).source_hash == "sha256:abc"


def test_context_pack_response_payload_adds_prompt_safe_context_text() -> None:
    pack = build_context_pack(
        [
            make_search_result(
                id="hit-1",
                text="Invoice payment context",
                document_id="billing-doc",
                document_key="billing.md",
                title="Billing",
                chunk_index=3,
            )
        ],
        query="billing",
    )

    payload = context_pack_response_payload(pack)

    assert payload["context_text"] == pack.as_prompt_text()
    assert payload["snippets"] == pack.to_payload()["snippets"]
    assert payload["char_count"] == pack.to_payload()["char_count"]
    assert payload["char_count"] != len(pack.as_prompt_text())
    assert payload["citation_summary"] == pack.to_payload()["citation_summary"]
    assert payload["citation_summary"] != pack.to_prompt_payload()["citation_summary"]


def test_context_pack_prompt_payload_uses_rank_local_citations() -> None:
    pack = build_context_pack(
        [
            make_search_result(
                id="private-hit-1",
                text="Invoice payment context",
                document_id="internal-doc-id",
                document_key="private/billing.md",
                title=None,
                collection="tenant-corpus",
                chunk_index=3,
                content_sha256="sha256:abc",
            )
        ],
        query="billing",
    )

    payload = pack.to_prompt_payload()
    encoded = str(payload)

    assert "[S1] file" in pack.as_prompt_text()
    assert "internal-doc-id" not in encoded
    assert "private/billing.md" not in encoded
    assert "private-hit-1" not in encoded
    assert "tenant-corpus" not in encoded
    assert "sha256:abc" not in encoded
    snippets = payload["snippets"]
    assert isinstance(snippets, list)
    snippet = snippets[0]
    assert isinstance(snippet, dict)
    assert snippet["citation_id"] == "S1"
    assert snippet["char_count"] == len(pack.snippets[0].as_prompt_text())
    assert snippet["char_count"] != pack.snippets[0].char_count
    source = snippet["source"]
    assert isinstance(source, dict)
    assert source == {
        "citation_id": "S1",
        "chunk_index": 3,
        "source_type": "file",
    }


def test_context_pack_prompt_payload_keeps_metadata_allowlisted() -> None:
    pack = build_context_pack(
        [
            make_search_result(
                id="private-hit-1",
                text="Invoice payment context",
                document_id="internal-doc-id",
                document_key="private/billing.md",
                collection="tenant-corpus",
                chunk_index=3,
                content_sha256="sha256:abc",
                metadata={
                    "quality_verdict": "poor",
                    "quality_char_count": 128,
                    "quality_ignored": "private/billing.md",
                    "rerank": {
                        "provider": "voyage",
                        "model": "rerank-2.5-lite",
                        "provider_score": 0.91,
                        "ignored": {"document_id": "internal-doc-id"},
                    },
                },
            )
        ],
        query="billing",
    )

    payload = pack.to_prompt_payload()
    encoded = str(payload)

    assert "internal-doc-id" not in encoded
    assert "private/billing.md" not in encoded
    assert "tenant-corpus" not in encoded
    assert "sha256:abc" not in encoded
    snippets = payload["snippets"]
    assert isinstance(snippets, list)
    snippet = snippets[0]
    assert isinstance(snippet, dict)
    assert snippet["retrieval_metadata"] == {
        "quality": {"verdict": "poor", "char_count": 128},
        "rerank": {
            "provider": "voyage",
            "model": "rerank-2.5-lite",
            "provider_score": 0.91,
        },
    }


def test_context_pack_prompt_payload_keeps_remote_url_identity_private() -> None:
    pack = build_context_pack(
        [
            make_search_result(
                id="remote-hit-1",
                text="Remote export context",
                source_type="url",
                document_id="internal-remote-doc",
                document_key="url:https://example.com/export?redacted|query_sha256:abc123",
                title="https://example.com/export?redacted",
                collection="tenant-corpus",
                chunk_index=2,
                content_sha256="sha256:remote",
            )
        ],
        query="remote export",
    )

    payload = pack.to_prompt_payload()
    encoded = str(payload) + pack.as_prompt_text()

    assert "https://example.com/export?redacted" in encoded
    assert "query_sha256" not in encoded
    assert "abc123" not in encoded
    assert "internal-remote-doc" not in encoded
    assert "tenant-corpus" not in encoded
    assert "sha256:remote" not in encoded
    snippets = payload["snippets"]
    assert isinstance(snippets, list)
    snippet = snippets[0]
    assert isinstance(snippet, dict)
    source = snippet["source"]
    assert isinstance(source, dict)
    assert source == {
        "citation_id": "S1",
        "title": "https://example.com/export?redacted",
        "chunk_index": 2,
        "source_type": "url",
    }
    previews = payload["source_previews"]
    assert isinstance(previews, list)
    preview = previews[0]
    assert isinstance(preview, dict)
    assert preview["title"] == "https://example.com/export?redacted"
    assert preview["source_type"] == "url"
    assert "document_key" not in preview


def test_source_locator_propagates_search_result_char_offsets() -> None:
    locator = source_locator_from_result(
        make_search_result(
            id="span-hit",
            start_char=120,
            end_char=240,
        )
    )

    assert locator.start_offset == 120
    assert locator.end_offset == 240


def test_evidence_span_from_result_yields_signal_desk_shape() -> None:
    span = evidence_span_from_result(
        make_search_result(
            id="span-hit",
            document_id="artifact-1",
            start_char=50,
            end_char=120,
            metadata={"page_number": 3},
        )
    )

    assert span == EvidenceSpan(
        artifact_id="artifact-1",
        start_offset=50,
        end_offset=120,
        page=3,
    )
    assert span.to_payload() == {
        "artifact_id": "artifact-1",
        "start_offset": 50,
        "end_offset": 120,
        "page": 3,
    }


def test_evidence_span_returns_none_when_offsets_missing() -> None:
    span = evidence_span_from_result(
        make_search_result(id="no-span", document_id="artifact-1")
    )

    assert span is None


def test_evidence_span_returns_none_when_document_id_missing() -> None:
    span = evidence_span_from_result(
        make_search_result(
            id="no-doc",
            document_id=None,
            start_char=0,
            end_char=10,
        )
    )

    assert span is None


def test_evidence_span_refuses_unreliable_offset_reconstruction() -> None:
    span = evidence_span_from_result(
        make_search_result(
            id="unreliable-span",
            document_id="artifact-1",
            start_char=50,
            end_char=120,
            metadata={"offset_reconstruction": "unreliable"},
        )
    )

    assert span is None


def test_source_locator_refuses_unreliable_offsets() -> None:
    # Unreliable flag: derived positional locators must not populate.
    locator_unreliable = source_locator_from_result(
        make_search_result(
            id="unreliable-loc",
            start_char=50,
            end_char=120,
            metadata={
                "offset_reconstruction": "unreliable",
                "page_number": 2,
                "page_index": 1,
                "line_start": 5,
                "line_end": 10,
                "figure_id": "fig-1",
                "figure_caption": "Synthetic figure",
                "figure_thumbnail_url": "https://cdn.example.com/fig.png",
            },
            figure_id="fig-1",
            figure_thumbnail_url="https://cdn.example.com/fig.png",
            chunk_index=1,
        )
    )
    assert locator_unreliable.start_offset is None
    assert locator_unreliable.end_offset is None
    assert locator_unreliable.page_number is None
    assert locator_unreliable.page_index is None
    assert locator_unreliable.line_start is None
    assert locator_unreliable.line_end is None
    assert locator_unreliable.figure_id is None
    assert locator_unreliable.figure_caption is None
    assert locator_unreliable.figure_thumbnail_url is None
    assert locator_unreliable.chunk_index == 1

    # No flag: offsets are present as normal.
    locator_reliable = source_locator_from_result(
        make_search_result(
            id="reliable-loc",
            start_char=50,
            end_char=120,
            metadata={"line_start": 5, "line_end": 10},
            chunk_index=1,
        )
    )
    assert locator_reliable.start_offset == 50
    assert locator_reliable.end_offset == 120


def test_context_pack_source_ids_do_not_expose_unreliable_figure_locators() -> None:
    pack = build_context_pack(
        [
            make_search_result(
                id="unreliable-figure",
                document_id="doc-1",
                metadata={
                    "offset_reconstruction": "unreliable",
                    "figure_id": "fig-1",
                },
                figure_id="fig-1",
                chunk_index=1,
            )
        ],
        query="figure",
    )

    assert pack.snippets[0].source.source_id == "space-1:corpus-1:doc-1#chunk-1"
    assert pack.snippets[0].locator.figure_id is None


def test_context_pack_truncation_narrows_locator_end_offset_to_kept_slice() -> None:
    long_text = "abcdefghijklmnopqrstuvwxyz" * 4  # 104 chars
    pack = build_context_pack(
        [
            make_search_result(
                id="long-hit",
                document_id="doc-long",
                chunk_index=0,
                text=long_text,
                start_char=1000,
                end_char=1000 + len(long_text),
            )
        ],
        query="truncation",
        max_chars=80,
    )

    [snippet] = pack.snippets
    assert snippet.truncated is True
    assert snippet.locator.start_offset == 1000
    # end_offset must shrink to match the kept slice, not the original chunk end.
    assert snippet.locator.end_offset is not None
    assert snippet.locator.end_offset == 1000 + len(snippet.text)
    assert snippet.locator.end_offset < 1000 + len(long_text)


def test_rag_core_retrieve_context_emits_context_pack_timing() -> None:
    async def scenario() -> EventBuffer:
        events = EventBuffer()
        core = Engine(
            make_test_config(
                qdrant_collection="rag_core_context_pack_events",
                embedding_dimensions=4,
            ),
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=RecordingVectorStore(
                search_results=[make_search_result(id="hit-1")]
            ),
            event_sink=events,
        )
        try:
            await core.context(
                query="How do I pay?",
                namespace="acme",
                collections=["help"],
                limit=4,
                rerank=False,
            )
        finally:
            await core.close()
        return events

    events = asyncio.run(scenario())
    context_events = [
        e
        for e in events.events
        if isinstance(e, SearchStageCompleted) and e.stage == "context_pack"
    ]
    [event] = context_events
    assert event.stage_name == "build_context_pack"
    assert event.candidate_count == 1
    assert event.result_count == 1
    assert event.duration_ms >= 0.0
