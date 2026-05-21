import asyncio

import pytest

from rag_core import ModelContextPack, RAGCore
from rag_core.events import EventBuffer
from rag_core.search.context_pack import build_context_pack
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


def test_context_pack_preserves_rank_order_and_source_references() -> None:
    first = make_search_result(
        id="point-1",
        text="Billing happens on the first day of each month.",
        score=0.91,
        document_id="billing-doc",
        corpus_id="help",
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
        corpus_id="help",
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
    assert pack.snippets[0].header == "[space-1:corpus-1:deck#chunk-0] Deck Slide 2, chunk 0"
    assert pack.snippets[1].locator.sheet_name == "Signals"
    assert pack.snippets[1].locator.row_range == "11-20"
    assert pack.snippets[1].header == "[space-1:corpus-1:workbook#chunk-1] Workbook sheet Signals, rows 11-20, chunk 1"
    assert pack.snippets[2].header == "[space-1:corpus-1:workbook#chunk-2] Workbook Sheet: Signals (Rows 11-20), chunk 2"
    snippets = pack.to_payload()["snippets"]
    assert isinstance(snippets, list)
    second = snippets[1]
    assert isinstance(second, dict)
    locator = second["locator"]
    assert isinstance(locator, dict)
    assert locator["sheet_name"] == "Signals"
    assert locator["page_number"] is None


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

    assert pack.snippets[0].header == "[space-1:corpus-1:doc#chunk-0] Document Page 4, chunk 0"


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
        "bbox",
        "figure_id",
        "figure_caption",
        "figure_thumbnail_url",
    }
    assert optional_keys.issubset(locator)
    assert all(locator[key] is None for key in optional_keys)


def test_context_pack_dedupes_repeated_source_chunks_and_fills_budget() -> None:
    first = make_search_result(
        id="point-1", document_id=None, document_key="guide.md", chunk_index=1, text="first guide chunk"
    )
    duplicate = make_search_result(
        id="point-2", document_id=None, document_key="guide.md", chunk_index=1, text="duplicate guide chunk"
    )
    second = make_search_result(
        id="point-3", document_id=None, document_key="guide.md", chunk_index=2, text="second guide chunk"
    )
    dropped_by_limit = make_search_result(
        id="point-4", document_id=None, document_key=None, chunk_index=None, text="fallback chunk"
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
    first = make_search_result(id="point-1", document_id="guide", chunk_index=None, text="first guide hit")
    second = make_search_result(id="point-2", document_id="guide", chunk_index=None, text="second guide hit")

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
        corpus_id=None,
        document_key="guide.md",
        chunk_index=0,
        text="first guide hit",
    )
    second = make_search_result(
        id="point-2",
        document_id=None,
        corpus_id=None,
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
        corpus_id=None,
        document_key="guide.md",
        chunk_index=0,
        text="first guide hit",
    )
    second = make_search_result(
        id="point-2",
        document_id=None,
        corpus_id=None,
        document_key="guide.md",
        chunk_index=0,
        text="second guide hit",
    )

    first_order = build_context_pack([first, second], query="guide")
    reversed_order = build_context_pack([second, first], query="guide")

    first_ids = {snippet.text: snippet.citation_id for snippet in first_order.snippets}
    reversed_ids = {snippet.text: snippet.citation_id for snippet in reversed_order.snippets}
    assert first_ids == reversed_ids
    assert first_ids["first guide hit"].startswith("space-1:guide.md#chunk-0-")
    assert first_ids["second guide hit"].startswith("space-1:guide.md#chunk-0-")
    assert first_ids["first guide hit"] != first_ids["second guide hit"]


def test_context_pack_citation_ids_are_corpus_scoped_when_available() -> None:
    first = make_search_result(
        id="point-1",
        corpus_id="corp-a",
        document_id="guide",
        chunk_index=1,
        text="first corpus hit",
    )
    second = make_search_result(
        id="point-2",
        corpus_id="corp-b",
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


def test_context_pack_keeps_section_citation_ids_stable_when_rank_order_changes() -> None:
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
    reversed_ids = {snippet.text: snippet.citation_id for snippet in reversed_order.snippets}
    assert first_ids == reversed_ids


def test_context_pack_enforces_char_budget_and_reports_drops() -> None:
    pack = build_context_pack(
        [
            make_search_result(id="a", text="abcdefghij", document_id="doc-a", chunk_index=0),
            make_search_result(id="b", text="klmnop", document_id="doc-b", chunk_index=0),
            make_search_result(id="c", text="qrstuv", document_id="doc-c", chunk_index=0),
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
        [make_search_result(id="a", text="abcdefghij", document_id="doc-a", chunk_index=0)],
        query="budget",
        max_chars=3,
    )

    assert pack.as_text() == ""
    assert pack.char_count == 0
    assert pack.dropped_count == 1
    assert pack.truncated is True


def test_context_pack_token_budget_translates_to_char_budget() -> None:
    pack = build_context_pack(
        [make_search_result(id="a", text="abcdefghij", document_id="doc-a", chunk_index=0)],
        query="budget",
        max_tokens=10,
        chars_per_token=2,
    )

    assert len(pack.as_text()) <= 20
    assert pack.max_chars == 20
    assert pack.char_count == len(pack.as_text())


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
    async def scenario() -> tuple[ModelContextPack, RecordingVectorStore]:
        store = RecordingVectorStore(
            search_results=[
                make_search_result(
                    id="hit-1",
                    text="Invoice payment context",
                    document_id="billing-doc",
                    corpus_id="help",
                    chunk_index=3,
                )
            ]
        )
        core = RAGCore(
            make_test_config(qdrant_collection="rag_core_context_pack", embedding_dimensions=4),
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=store,
        )
        try:
            pack = await core.retrieve_context(
                query="How do I pay?",
                namespace="acme",
                corpus_ids=["help"],
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
    assert store.search_calls[0].corpus_ids == ["help"]


def test_rag_core_retrieve_context_honors_explicit_query_plan_final_limit() -> None:
    async def scenario() -> tuple[ModelContextPack, RecordingVectorStore]:
        store = RecordingVectorStore(
            search_results=[
                make_search_result(id="hit-1", text="One", document_id="doc-1", chunk_index=0),
                make_search_result(id="hit-2", text="Two", document_id="doc-2", chunk_index=0),
                make_search_result(id="hit-3", text="Three", document_id="doc-3", chunk_index=0),
            ]
        )
        core = RAGCore(
            make_test_config(
                qdrant_collection="rag_core_context_pack_query_plan_limit",
                embedding_dimensions=4,
            ),
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=store,
        )
        try:
            pack = await core.retrieve_context(
                query="How do I pay?",
                namespace="acme",
                corpus_ids=["help"],
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


def test_rag_core_retrieve_context_emits_context_pack_timing() -> None:
    async def scenario() -> EventBuffer:
        events = EventBuffer()
        core = RAGCore(
            make_test_config(
                qdrant_collection="rag_core_context_pack_events",
                embedding_dimensions=4,
            ),
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=RecordingVectorStore(search_results=[make_search_result(id="hit-1")]),
            event_sink=events,
        )
        try:
            await core.retrieve_context(
                query="How do I pay?",
                namespace="acme",
                corpus_ids=["help"],
                limit=4,
                rerank=False,
            )
        finally:
            await core.close()
        return events

    events = asyncio.run(scenario())
    context_events = [
        e for e in events.events
        if isinstance(e, SearchStageCompleted) and e.stage == "context_pack"
    ]
    [event] = context_events
    assert event.stage_name == "build_context_pack"
    assert event.candidate_count == 1
    assert event.result_count == 1
    assert event.duration_ms >= 0.0
