from __future__ import annotations

import asyncio
from dataclasses import fields

import pytest

from rag_core import SearchOptions
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


def test_common_search_options_expose_explicit_modes() -> None:
    option_fields = {field.name for field in fields(SearchOptions)}

    assert "mode" in option_fields
    assert SearchOptions().mode == "dense"
    assert SearchOptions(mode="lexical").mode == "lexical"
    assert SearchOptions(mode="hybrid").mode == "hybrid"


def test_common_search_options_expose_metadata_filter() -> None:
    assert "metadata_filter" in {field.name for field in fields(SearchOptions)}


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
