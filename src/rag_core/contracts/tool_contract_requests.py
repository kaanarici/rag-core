from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .tool_contract_schemas import (
    SEARCH_USER_DOCUMENTS_DEFAULT_LIMIT,
    SEARCH_USER_DOCUMENTS_DEFAULT_MAX_CHARS,
    SEARCH_USER_DOCUMENTS_DEFAULT_RERANK,
    SEARCH_USER_DOCUMENTS_DEFAULT_USE_LEXICAL_SEARCH,
    SEARCH_USER_DOCUMENTS_LIMIT_MAX,
    SEARCH_USER_DOCUMENTS_LIMIT_MIN,
    SEARCH_USER_DOCUMENTS_MAX_CHARS_MAX,
    SEARCH_USER_DOCUMENTS_MAX_CHARS_MIN,
    SEARCH_USER_DOCUMENTS_MAX_TOKENS_MAX,
    SEARCH_USER_DOCUMENTS_MAX_TOKENS_MIN,
    _SEARCH_USER_DOCUMENTS_INPUT_FIELDS,
)


@dataclass(frozen=True)
class SearchUserDocumentsRequest:
    query: str
    limit: int = SEARCH_USER_DOCUMENTS_DEFAULT_LIMIT
    document_ids: tuple[str, ...] | None = None
    rerank: bool = SEARCH_USER_DOCUMENTS_DEFAULT_RERANK
    use_lexical_search: bool = SEARCH_USER_DOCUMENTS_DEFAULT_USE_LEXICAL_SEARCH
    max_chars: int | None = SEARCH_USER_DOCUMENTS_DEFAULT_MAX_CHARS
    max_tokens: int | None = None


def parse_search_user_documents_request(
    payload: Mapping[str, object],
    *,
    default_limit: int = SEARCH_USER_DOCUMENTS_DEFAULT_LIMIT,
    default_rerank: bool = SEARCH_USER_DOCUMENTS_DEFAULT_RERANK,
    default_use_lexical_search: bool = SEARCH_USER_DOCUMENTS_DEFAULT_USE_LEXICAL_SEARCH,
    default_max_chars: int | None = SEARCH_USER_DOCUMENTS_DEFAULT_MAX_CHARS,
    default_max_tokens: int | None = None,
) -> SearchUserDocumentsRequest:
    """Parse and validate ``search_user_documents`` input."""
    if set(payload) - _SEARCH_USER_DOCUMENTS_INPUT_FIELDS:
        raise ValueError("search_user_documents input contains unsupported fields")

    query = _required_non_empty_string(payload, "query")
    limit = _optional_int(
        payload,
        "limit",
        default=default_limit,
    )
    if limit is None:
        raise ValueError("limit must be an integer")
    max_chars = _optional_int(
        payload,
        "max_chars",
        default=default_max_chars,
    )
    max_tokens = _optional_int(payload, "max_tokens", default=default_max_tokens)
    validate_search_user_documents_bounds(
        limit=limit,
        max_chars=max_chars,
        max_tokens=max_tokens,
    )
    return SearchUserDocumentsRequest(
        query=query,
        limit=limit,
        document_ids=_optional_document_ids(payload),
        rerank=_optional_bool(
            payload,
            "rerank",
            default=default_rerank,
        ),
        use_lexical_search=_optional_bool(
            payload,
            "use_lexical_search",
            default=default_use_lexical_search,
        ),
        max_chars=max_chars,
        max_tokens=max_tokens,
    )


def validate_search_user_documents_bounds(
    *,
    limit: int | None = None,
    max_chars: int | None = None,
    max_tokens: int | None = None,
) -> None:
    """Validate ``search_user_documents`` bounds before retrieval."""
    if limit is not None and not (SEARCH_USER_DOCUMENTS_LIMIT_MIN <= limit <= SEARCH_USER_DOCUMENTS_LIMIT_MAX):
        raise ValueError(
            f"limit must be between {SEARCH_USER_DOCUMENTS_LIMIT_MIN} and {SEARCH_USER_DOCUMENTS_LIMIT_MAX}"
        )
    if max_chars is not None and not (
        SEARCH_USER_DOCUMENTS_MAX_CHARS_MIN <= max_chars <= SEARCH_USER_DOCUMENTS_MAX_CHARS_MAX
    ):
        raise ValueError(
            "max_chars must be between "
            f"{SEARCH_USER_DOCUMENTS_MAX_CHARS_MIN} and {SEARCH_USER_DOCUMENTS_MAX_CHARS_MAX}"
        )
    if max_tokens is not None and not (
        SEARCH_USER_DOCUMENTS_MAX_TOKENS_MIN <= max_tokens <= SEARCH_USER_DOCUMENTS_MAX_TOKENS_MAX
    ):
        raise ValueError(
            "max_tokens must be between "
            f"{SEARCH_USER_DOCUMENTS_MAX_TOKENS_MIN} and {SEARCH_USER_DOCUMENTS_MAX_TOKENS_MAX}"
        )


def _required_non_empty_string(payload: Mapping[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_int(
    payload: Mapping[str, object],
    field: str,
    *,
    default: int | None,
) -> int | None:
    if field not in payload:
        return default
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _optional_bool(
    payload: Mapping[str, object],
    field: str,
    *,
    default: bool,
) -> bool:
    if field not in payload:
        return default
    value = payload[field]
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _optional_document_ids(payload: Mapping[str, object]) -> tuple[str, ...] | None:
    if "document_ids" not in payload:
        return None
    value = payload["document_ids"]
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError("document_ids must be an array of non-empty strings")
    ids: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("document_ids must be an array of non-empty strings")
        ids.append(item.strip())
    return tuple(ids)
