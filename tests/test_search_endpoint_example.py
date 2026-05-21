from __future__ import annotations

import asyncio

import pytest

from examples.search_endpoint import search_user_documents_endpoint, seed_demo_corpus
from rag_core.demo import build_demo_core


def test_search_endpoint_example_returns_tool_result_payload() -> None:
    payload = asyncio.run(_run_endpoint_payload(authorized_document_ids=["billing"]))

    assert payload["ok"] is True
    assert "context_text" in payload
    assert "billing" in str(payload["context_text"]).lower()
    assert payload["max_snippets"] == 3


def test_search_endpoint_example_requires_app_authorization_for_document_filters() -> None:
    with pytest.raises(ValueError, match="authorized_document_ids must be bound"):
        asyncio.run(
            _run_endpoint_payload(
                {
                    "query": "How can a customer pay an invoice?",
                    "document_ids": ["billing"],
                    "rerank": False,
                }
            )
        )


def test_search_endpoint_example_applies_authorized_document_filters() -> None:
    payload = asyncio.run(
        _run_endpoint_payload(
            {
                "query": "How can a customer pay an invoice?",
                "document_ids": ["billing"],
                "rerank": False,
            },
            authorized_document_ids=["billing"],
        )
    )

    context_text = str(payload["context_text"]).lower()
    assert "billing" in context_text
    assert "shipping" not in context_text


def test_search_endpoint_example_defaults_to_authorized_scope_when_document_ids_omitted() -> None:
    payload = asyncio.run(
        _run_endpoint_payload(
            {
                "query": "How long does shipping take?",
                "rerank": False,
                "max_chars": 1200,
            },
            authorized_document_ids=["billing"],
        )
    )

    context_text = str(payload["context_text"]).lower()
    assert "shipping" not in context_text
    citations = payload["citations"]
    assert isinstance(citations, list)
    assert all(
        "shipping" not in str(citation.get("source_id", "")).lower()
        for citation in citations
    )


def test_search_endpoint_example_empty_authorized_scope_returns_no_context() -> None:
    payload = asyncio.run(
        _run_endpoint_payload(
            {
                "query": "How long does shipping take?",
                "rerank": False,
                "max_chars": 1200,
            },
            authorized_document_ids=[],
        )
    )

    assert payload["ok"] is True
    assert payload["snippets"] == []
    assert payload["citations"] == []
    assert payload["context_text"] == ""


async def _run_endpoint_payload(
    payload: dict[str, object] | None = None,
    *,
    authorized_document_ids: list[str] | None = None,
) -> dict[str, object]:
    core = build_demo_core(collection="search_endpoint_test")
    try:
        await core.ensure_ready()
        await seed_demo_corpus(core)
        return await search_user_documents_endpoint(
            core,
            payload
            or {
                "query": "How can a customer pay an invoice?",
                "limit": 3,
                "rerank": False,
                "max_chars": 1200,
            },
            namespace="acme",
            corpus_ids=["help-center"],
            authorized_document_ids=authorized_document_ids,
        )
    finally:
        await core.close()
