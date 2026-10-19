"""Review regressions: local Qdrant integration and payload/trace/context contracts.

Deterministic vectors establish storage/filter plumbing, not retrieval quality.
"""

from __future__ import annotations

import asyncio
import math
import uuid
from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Any

import pytest

from rag_core.core_models import PreparedChunk
from rag_core.events.sink_payloads import event_to_jsonl_dict
from rag_core.events.traces import (
    search_event_from_payload,
    summarize_search_trace_payload_runs,
    summarize_search_trace_payloads,
    summarize_search_trace_runs,
)
from rag_core.events.types import Event, SearchCompleted, SearchStarted, StageError
from rag_core.search.context_pack import build_context_pack
from rag_core.search.filters import And, Filter, In, Not, Or, Term
from rag_core.search.indexer import DocumentIndexer, IndexRequest
from rag_core.search.policy import VectorStorePolicy
from rag_core.search.request_models import SearchQuery
from rag_core.search.result_filters import result_matches_query
from rag_core.search.stored_payload import build_stored_payload, payload_to_result
from rag_core.search.vector_models import ContentType, SearchResult, VectorPoint
from tests.support import FakeEmbeddingProvider
from tests.support.local_store import LocalQdrantStore


@pytest.mark.integration
@pytest.mark.parametrize(
    "metadata_filter,expected",
    [
        (Term("price", 1.5), {"fraction"}),
        (Term("price", 2.0), {"integer"}),
        (In("price", (1.5,)), {"fraction"}),
        (In("price", (1.5, 2.5)), {"fraction", "other"}),
        (In("price", (1.5, 2, "free")), {"fraction", "integer", "string"}),
        (In("price", (2,)), {"integer"}),
        (In("price", ("free",)), {"string"}),
        (And((Term("price", 1.5), Term("enabled", True))), {"fraction"}),
        (Or((Term("price", 1.5), Term("price", 2.5))), {"fraction", "other"}),
        (Not(In("price", (1.5, 2.5))), {"integer", "string"}),
    ],
)
def test_qdrant_numeric_equality_and_mixed_membership(
    metadata_filter: Filter, expected: set[str]
) -> None:
    async def run() -> None:
        store = LocalQdrantStore()
        points = []
        for tenant, name, price in (
            ("tenant", "fraction", 1.5),
            ("tenant", "integer", 2),
            ("tenant", "other", 2.5),
            ("tenant", "string", "free"),
            ("elsewhere", "foreign", 1.5),
        ):
            points.append(
                VectorPoint(
                    id=str(uuid.uuid4()),
                    dense_vector=[1.0, 0.0, 0.0, 0.0],
                    payload={
                        "tenant_id": tenant,
                        "collection": "docs",
                        "document_id": name,
                        "text": name,
                        "content_type": "document",
                        "enabled": True,
                        "price": price,
                    },
                )
            )
        try:
            await store.upsert(points)
            hits = await store.search(
                SearchQuery(
                    dense_vector=[1.0, 0.0, 0.0, 0.0],
                    tenant_id="tenant",
                    collections=["docs"],
                    metadata_filter=metadata_filter,
                )
            )
            assert {hit.document_id for hit in hits} == expected
        finally:
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("alias", ["text_field", "tenant_id_field", "title_field"])
@pytest.mark.parametrize(
    "fixed_field",
    [
        "mime_type",
        "section_id",
        "page_number",
        "start_char",
        "ingest_fingerprint",
        "_rag_revision",
    ],
)
def test_policy_refuses_aliases_that_overwrite_fixed_payloads(
    alias: str, fixed_field: str
) -> None:
    fields: dict[str, Any] = {alias: fixed_field}
    with pytest.raises(ValueError, match="reserved document metadata"):
        replace(VectorStorePolicy(), **fields)


def test_quality_projection_cannot_overwrite_a_policy_alias() -> None:
    policy = VectorStorePolicy(text_field="quality_verdict")
    payload = build_stored_payload(
        tenant_id="tenant",
        collection="docs",
        document_id="doc",
        document_key=None,
        content_sha256=None,
        processing_version=None,
        filename="file.md",
        mime_type="text/markdown",
        source_type="file",
        document_path=None,
        chunk_index=0,
        payload_text="body",
        content_type=ContentType.DOCUMENT,
        chunker_strategy="markdown",
        title=None,
        filter_metadata={"quality": {"verdict": "poor"}},
        policy=policy,
    )
    assert (
        payload_to_result(point_id="p", payload=payload, score=1, policy=policy).text
        == "body"
    )


@pytest.mark.integration
@pytest.mark.parametrize("immutable", [False, True])
def test_indexer_preserves_canonical_mapping_metadata(immutable: bool) -> None:
    async def run() -> None:
        metadata: Mapping[str, object] = {"team": "support", "page_number": 2}
        if immutable:
            metadata = MappingProxyType(dict(metadata))
        store = LocalQdrantStore()
        try:
            indexer = DocumentIndexer(FakeEmbeddingProvider(), None, store)
            await indexer.index_document(
                IndexRequest(
                    document_id="doc",
                    collection="docs",
                    tenant_id="tenant",
                    text="body",
                    filename="file.md",
                    mime_type="text/markdown",
                    source_type="file",
                    document_metadata={"team": "document", "region": "west"},
                    prepared_chunks=[
                        PreparedChunk(
                            chunk_index=0,
                            text="body",
                            embedding_text="body",
                            word_count=1,
                            metadata=metadata,
                        )
                    ],
                )
            )
            [hit] = await store.search(
                SearchQuery(
                    dense_vector=[0.1, 0.2, 0.3, 0.4],
                    tenant_id="tenant",
                    collections=["docs"],
                    metadata_filter=Term("team", "support"),
                )
            )
            assert hit.metadata["team"] == "support"
            assert hit.metadata["region"] == "west"
            assert hit.metadata["page_number"] == 2
            assert metadata == {"team": "support", "page_number": 2}
        finally:
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("search_ids", [("search-a",), ("search-a", "search-b"), ("",)])
@pytest.mark.parametrize("ingest_id", ["ingest-a", "", "private/ingest"])
def test_ingest_errors_stay_outside_typed_and_jsonl_search_summaries(
    search_ids: tuple[str, ...],
    ingest_id: str,
) -> None:
    events: list[Event] = [
        StageError(ingest_id=ingest_id, stage="parse", error_type="ValueError")
    ]
    for search_id in search_ids:
        events.extend(
            [
                SearchStarted(search_id=search_id, query_length=7),
                StageError(
                    search_id=search_id, stage="rerank", error_type="TimeoutError"
                ),
                SearchCompleted(search_id=search_id, result_count=1, succeeded=True),
            ]
        )
    typed = summarize_search_trace_runs(events)
    payloads = [event_to_jsonl_dict(event) for event in events]
    assert summarize_search_trace_payload_runs(payloads) == typed
    assert len(typed) == len(search_ids)
    for summary in typed:
        assert summary.succeeded and summary.completed
        assert summary.error_stages == ("rerank",)
        assert summary.error_types == ("TimeoutError",)
    if len(search_ids) == 1:
        assert summarize_search_trace_payloads(payloads) == typed[0]
    assert summarize_search_trace_runs(events[:1]) == ()
    assert summarize_search_trace_payload_runs(payloads[:1]) == ()


def test_search_error_retains_search_correlation_when_ingest_id_is_also_present() -> (
    None
):
    event = StageError(
        search_id="search-a",
        ingest_id="ingest-a",
        stage="retrieve",
        error_type="ValueError",
    )
    decoded = search_event_from_payload(event_to_jsonl_dict(event))
    assert decoded == event
    [summary] = summarize_search_trace_runs([event])
    assert summary.search_id == "search-a"
    assert summary.error_stages == ("retrieve",)
    assert "ingest-a" not in repr(summary.to_payload())


def test_scope_guard_requires_backend_metadata_filtering() -> None:
    hit = SearchResult(
        id="x",
        text="body",
        score=1,
        content_type="document",
        source_type="file",
        tenant_id="tenant",
        collection="docs",
        document_id="doc",
    )
    query = SearchQuery(dense_vector=[1.0], tenant_id="tenant", collections=["docs"])
    assert result_matches_query(hit, query)
    assert not result_matches_query(replace(hit, tenant_id="other"), query)
    assert not result_matches_query(
        replace(hit, document_id="other"), replace(query, document_ids=["doc"])
    )
    assert not result_matches_query(hit, replace(query, collections=[]))
    with pytest.raises(ValueError, match="applied by the vector store"):
        result_matches_query(hit, replace(query, metadata_filter=Term("text", "body")))


@pytest.mark.parametrize(
    "max_chars,expected", [(10, None), (14, None), (16, "ab"), (24, "abcdefghij")]
)
def test_prompt_header_is_budgeted_before_evidence(
    max_chars: int, expected: str | None
) -> None:
    hit = SearchResult(
        id="x",
        text="abcdefghij",
        score=1,
        content_type="document",
        source_type="document",
        start_char=5,
        end_char=15,
    )
    pack = build_context_pack([hit], query="q", max_chars=max_chars)
    if expected is None:
        assert pack.snippets == ()
        assert pack.dropped_count == 1 and pack.truncated
        assert pack.as_prompt_text() == pack.as_text() == ""
    else:
        [snippet] = pack.snippets
        assert snippet.text == expected
        assert pack.as_prompt_text() == f"[S1] document\n{expected}"
        assert pack.as_text() == f"[x] x\n{expected}"
        assert snippet.locator.end_offset == 5 + len(expected)
        assert snippet.char_count == len(snippet.as_text())
        assert pack.to_prompt_payload()["char_count"] == len(pack.as_prompt_text())
    assert len(pack.as_text()) <= max_chars
    assert len(pack.as_prompt_text()) <= max_chars


@pytest.mark.parametrize("max_tokens", [9, 43, 100, 1000])
def test_both_context_projections_fit_whole_snippets_and_token_estimates(
    max_tokens: int,
) -> None:
    hits = [
        SearchResult(
            id=f"x{i}",
            text="evidence " * 4,
            score=1,
            content_type="document",
            source_type="document",
            document_id=(f"private-document-{i}" if i % 2 else None),
        )
        for i in range(12)
    ]
    pack = build_context_pack(
        hits,
        query="q",
        max_snippets=12,
        max_chars=1000,
        max_tokens=max_tokens,
        chars_per_token=3,
    )
    assert pack.max_chars == min(1000, max_tokens * 3)
    if max_tokens == 1000:
        assert len(pack.snippets) == 12
    for order in ("rank", "extrema"):
        prompt = pack.as_prompt_text(context_order=order)
        assert len(prompt) <= pack.max_chars
        assert "private-document" not in prompt
        for snippet in pack.snippets:
            assert snippet.as_prompt_text() in prompt
            assert snippet.as_text() in pack.as_text()
    assert pack.char_count == len(pack.as_text()) <= pack.max_chars
    assert pack.token_estimate == math.ceil(
        max(len(pack.as_text()), len(pack.as_prompt_text())) / 3
    )
    assert pack.token_estimate <= max_tokens
