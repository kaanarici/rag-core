from __future__ import annotations

import asyncio

import pytest

from rag_core import RAGCore
from rag_core.contracts import (
    SEARCH_USER_DOCUMENTS_INPUT_SCHEMA,
    SearchUserDocumentsRequest,
    parse_search_user_documents_request,
)
from rag_core.core_retrieval import search_with_core
from rag_core.search.searcher import SearchRequest
from rag_core.search.types import SearchResult


def test_search_user_documents_schema_uses_lexical_search_language() -> None:
    properties = SEARCH_USER_DOCUMENTS_INPUT_SCHEMA["properties"]

    assert isinstance(properties, dict)
    assert "use_lexical_search" in properties
    assert "use_sidecar" not in properties
    lexical_property = properties["use_lexical_search"]
    assert isinstance(lexical_property, dict)
    assert lexical_property["default"] is True


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
    search_parameters = RAGCore.search.__annotations__
    retrieve_parameters = RAGCore.retrieve_context.__annotations__

    assert "use_lexical_search" in search_parameters
    assert "use_sidecar" not in search_parameters
    assert "use_lexical_search" in retrieve_parameters
    assert "use_sidecar" not in retrieve_parameters


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
            corpus_ids=["help-center"],
            use_lexical_search=False,
        )
    )

    assert search.requests[0].use_sidecar is False
