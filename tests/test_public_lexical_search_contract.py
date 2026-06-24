from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from rag_core import Engine
from rag_core.contracts import (
    SEARCH_USER_DOCUMENTS_INPUT_SCHEMA,
    SearchUserDocumentsRequest,
    parse_search_user_documents_request,
)
from rag_core._engine.core_retrieval import search_with_core
from rag_core.search.pipeline_runner import SearchRequest
from rag_core.search.vector_models import SearchResult


def test_search_user_documents_schema_uses_lexical_search_language() -> None:
    properties = SEARCH_USER_DOCUMENTS_INPUT_SCHEMA["properties"]

    assert isinstance(properties, dict)
    assert "use_lexical_search" in properties
    assert "use_sidecar" not in properties
    lexical_property = properties["use_lexical_search"]
    assert isinstance(lexical_property, dict)
    assert lexical_property["default"] is True
    assert lexical_property["description"] == (
        "Controls configured lexical/exact-match expansion only; "
        "query-plan defaults remain provider capability-aware."
    )


def test_embed_docs_keep_lexical_sidecar_and_profile_separate() -> None:
    docs = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in (
            "docs-site/content/docs/search.mdx",
            "docs-site/content/docs/stability.mdx",
        )
    )
    normalized = " ".join(docs.split())

    # The `lexical` search profile (sparse BM25) is documented as a distinct
    # concept from the `use_lexical_search` request flag (the exact-match
    # sidecar), not collapsed or labeled a compatibility flag.
    assert "`use_lexical_search` is the request flag" in docs
    assert "compatibility flag" not in docs
    assert "The `lexical` profile means sparse BM25 retrieval" in normalized
    assert "gates a separate exact-match\n  sidecar" in docs or (
        "gates a separate exact-match sidecar" in normalized
    )


def test_parse_search_user_documents_request_rejects_sidecar_alias() -> None:
    request = parse_search_user_documents_request(
        {
            "query": " billing policy ",
            "use_lexical_search": False,
        }
    )

    assert request == SearchUserDocumentsRequest(
        query="billing policy",
        use_lexical_search=False,
    )

    with pytest.raises(ValueError, match="unsupported fields"):
        parse_search_user_documents_request(
            {
                "query": "billing policy",
                "use_sidecar": False,
            }
        )


def test_rag_core_facade_exposes_lexical_search_request_knob() -> None:
    search_parameters = Engine.search.__annotations__
    retrieve_parameters = Engine.context.__annotations__

    assert "use_lexical_search" in search_parameters
    assert "use_sidecar" not in search_parameters
    assert "use_lexical_search" in retrieve_parameters
    assert "use_sidecar" not in retrieve_parameters


def test_rag_core_facade_exposes_content_type_filter() -> None:
    search_parameters = Engine.search.__annotations__
    retrieve_parameters = Engine.context.__annotations__

    assert "content_types" in search_parameters
    assert "content_types" in retrieve_parameters


def test_core_retrieval_bridges_lexical_request_to_internal_sidecar_flag() -> None:
    class _Search:
        def __init__(self) -> None:
            self.requests: list[SearchRequest] = []

        async def search(self, req: SearchRequest) -> list[SearchResult]:
            self.requests.append(req)
            return []

    search = _Search()

    asyncio.run(
        search_with_core(
            search=search,
            query="billing",
            namespace="acme",
            collections=["help-center"],
            use_lexical_search=False,
        )
    )

    assert search.requests[0].execution.use_lexical_search is False


def test_core_retrieval_bridges_content_types_to_search_request() -> None:
    class _Search:
        def __init__(self) -> None:
            self.requests: list[SearchRequest] = []

        async def search(self, req: SearchRequest) -> list[SearchResult]:
            self.requests.append(req)
            return []

    search = _Search()

    asyncio.run(
        search_with_core(
            search=search,
            query="billing",
            namespace="acme",
            collections=["help-center"],
            content_types=["document"],
        )
    )

    assert search.requests[0].content_types == ["document"]
