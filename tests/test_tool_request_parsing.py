from __future__ import annotations

import pytest

from rag_core.contracts import (
    SEARCH_USER_DOCUMENTS_DEFAULT_LIMIT,
    SEARCH_USER_DOCUMENTS_DEFAULT_MAX_CHARS,
    SEARCH_USER_DOCUMENTS_INPUT_SCHEMA,
    SearchUserDocumentsRequest,
    normalize_static_retrieval_scope,
    parse_search_user_documents_request,
    scope_document_ids,
    validate_bound_namespace,
)


def test_parse_search_user_documents_request_applies_contract_defaults() -> None:
    request = parse_search_user_documents_request({"query": " billing policy "})

    assert request == SearchUserDocumentsRequest(
        query="billing policy",
        limit=SEARCH_USER_DOCUMENTS_DEFAULT_LIMIT,
        document_ids=None,
        rerank=False,
        use_lexical_search=True,
        max_chars=SEARCH_USER_DOCUMENTS_DEFAULT_MAX_CHARS,
        max_tokens=None,
    )


def test_parse_search_user_documents_request_accepts_bounded_fields() -> None:
    request = parse_search_user_documents_request(
        {
            "query": "refund status",
            "limit": 3,
            "document_ids": [" doc-1 ", "doc-2"],
            "rerank": False,
            "use_lexical_search": False,
            "max_chars": 512,
            "max_tokens": 128,
        }
    )

    assert request == SearchUserDocumentsRequest(
        query="refund status",
        limit=3,
        document_ids=("doc-1", "doc-2"),
        rerank=False,
        use_lexical_search=False,
        max_chars=512,
        max_tokens=128,
    )


def test_parse_search_user_documents_request_accepts_endpoint_defaults() -> None:
    request = parse_search_user_documents_request(
        {"query": "refund status"},
        default_limit=7,
        default_rerank=False,
        default_use_lexical_search=False,
        default_max_chars=None,
        default_max_tokens=256,
    )

    assert request == SearchUserDocumentsRequest(
        query="refund status",
        limit=7,
        document_ids=None,
        rerank=False,
        use_lexical_search=False,
        max_chars=None,
        max_tokens=256,
    )


def test_retrieval_scope_helpers_normalize_app_bound_scope() -> None:
    assert validate_bound_namespace(" acme ") == "acme"
    assert normalize_static_retrieval_scope(
        corpus_ids=[" help ", "docs"],
        document_ids=[" doc-1 "],
        limit=2,
    ) == (("help", "docs"), ("doc-1",))
    assert scope_document_ids(requested=None, configured=("doc-1", "doc-2")) == [
        "doc-1",
        "doc-2",
    ]
    assert scope_document_ids(requested=("doc-2",), configured=("doc-1", "doc-2")) == [
        "doc-2"
    ]


def test_retrieval_scope_helpers_reject_unbound_or_out_of_scope_values() -> None:
    with pytest.raises(ValueError, match="namespace must not be empty"):
        validate_bound_namespace(" ")
    with pytest.raises(ValueError, match="corpus_ids must not be empty"):
        normalize_static_retrieval_scope(corpus_ids=[], document_ids=None, limit=1)
    with pytest.raises(ValueError, match="document_ids must contain non-empty strings"):
        normalize_static_retrieval_scope(
            corpus_ids=["help"],
            document_ids=[" "],
            limit=1,
        )
    with pytest.raises(ValueError, match="outside the configured retrieval scope"):
        scope_document_ids(requested=("doc-2",), configured=("doc-1",))


def test_input_schema_rejects_blank_after_trim_values() -> None:
    properties = SEARCH_USER_DOCUMENTS_INPUT_SCHEMA["properties"]
    assert isinstance(properties, dict)

    query_schema = properties["query"]
    document_ids_schema = properties["document_ids"]
    assert isinstance(query_schema, dict)
    assert isinstance(document_ids_schema, dict)
    assert query_schema["pattern"] == r"\S"

    items = document_ids_schema["items"]
    assert isinstance(items, dict)
    assert items["pattern"] == r"\S"


def test_parse_search_user_documents_request_rejects_unknown_fields_without_echoing_payload() -> None:
    with pytest.raises(ValueError) as exc_info:
        parse_search_user_documents_request(
            {"query": "billing", "api_key_secret": "sk-test-secret"}
        )

    error = str(exc_info.value)
    assert error == "search_user_documents input contains unsupported fields"
    assert "api_key_secret" not in error
    assert "sk-test-secret" not in error


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"query": ""}, "query must be a non-empty string"),
        ({"query": "billing", "max_tokens": None}, "max_tokens must be an integer"),
        ({"query": "billing", "limit": True}, "limit must be an integer"),
        ({"query": "billing", "limit": 0}, "limit must be between 1 and 20"),
        (
            {"query": "billing", "document_ids": [""]},
            "document_ids must be an array of non-empty strings",
        ),
        (
            {"query": "billing", "document_ids": None},
            "document_ids must be an array of non-empty strings",
        ),
        ({"query": "billing", "rerank": "yes"}, "rerank must be a boolean"),
        ({"query": "billing", "max_chars": 128}, "max_chars must be between 256 and 12000"),
    ],
)
def test_parse_search_user_documents_request_rejects_invalid_shapes(
    payload: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError) as exc_info:
        parse_search_user_documents_request(payload)

    assert str(exc_info.value) == message
