from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rag_core.scope import resolve_collections_argument
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


def validate_limit_bounds(
    value: int | None,
    *,
    minimum: int,
    maximum: int,
    field: str,
) -> None:
    """Enforce ``minimum <= value <= maximum`` on a retrieval bound.

    Shared validator: the tool-contract entry points and the optional HTTP
    runtime both call this so bound enforcement stays identical.
    """
    if value is None:
        return
    if not (minimum <= value <= maximum):
        raise ValueError(
            f"{field} must be between {minimum} and {maximum}"
        )


def validate_search_user_documents_bounds(
    *,
    limit: int | None = None,
    max_chars: int | None = None,
    max_tokens: int | None = None,
) -> None:
    """Validate ``search_user_documents`` bounds before retrieval."""
    validate_limit_bounds(
        limit,
        minimum=SEARCH_USER_DOCUMENTS_LIMIT_MIN,
        maximum=SEARCH_USER_DOCUMENTS_LIMIT_MAX,
        field="limit",
    )
    validate_limit_bounds(
        max_chars,
        minimum=SEARCH_USER_DOCUMENTS_MAX_CHARS_MIN,
        maximum=SEARCH_USER_DOCUMENTS_MAX_CHARS_MAX,
        field="max_chars",
    )
    validate_limit_bounds(
        max_tokens,
        minimum=SEARCH_USER_DOCUMENTS_MAX_TOKENS_MIN,
        maximum=SEARCH_USER_DOCUMENTS_MAX_TOKENS_MAX,
        field="max_tokens",
    )


def validate_bound_namespace(namespace: str) -> str:
    """Normalize the app-bound tenant namespace before retrieval."""
    normalized = namespace.strip()
    if not normalized:
        raise ValueError("namespace must not be empty")
    return normalized


def normalize_static_retrieval_scope(
    *,
    collection: str | None = None,
    collections: Sequence[str] | None = None,
    document_ids: Sequence[str] | None,
    limit: int,
) -> tuple[tuple[str, ...], tuple[str, ...] | None]:
    """Normalize app-bound collection and document scope."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    resolved_collections = resolve_collections_argument(
        collection=collection,
        collections=collections,
        caller="normalize_static_retrieval_scope",
    )
    collections_tuple = _normalize_scope_values(resolved_collections, "collections")
    if not collections_tuple:
        raise ValueError("collections must not be empty")
    document_ids_tuple = (
        _normalize_scope_values(document_ids, "document_ids")
        if document_ids is not None
        else None
    )
    return collections_tuple, document_ids_tuple


def normalize_static_content_types(
    content_types: Sequence[str] | None,
) -> tuple[str, ...] | None:
    """Normalize app-bound content-type scope."""
    if content_types is None:
        return None
    return _normalize_scope_values(content_types, "content_types")


def scope_document_ids(
    *,
    requested: tuple[str, ...] | None,
    configured: tuple[str, ...] | None,
) -> list[str] | None:
    """Apply model-requested document IDs inside the app-bound document scope."""
    if configured is None:
        return list(requested) if requested is not None else None
    if requested is None:
        return list(configured)
    configured_set = set(configured)
    rejected = [document_id for document_id in requested if document_id not in configured_set]
    if rejected:
        raise ValueError("document_ids contain values outside the configured retrieval scope")
    return list(requested)


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


def _normalize_scope_values(values: Sequence[str], field: str) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must contain non-empty strings")
        normalized.append(value.strip())
    return tuple(normalized)
