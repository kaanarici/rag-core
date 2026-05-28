"""Typed request parsing for the optional HTTP runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from rag_core.retrieval_defaults import (
    DEFAULT_CONTEXT_LIMIT,
    DEFAULT_RERANK,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_USE_LEXICAL_SEARCH,
)
from rag_core.runtime.errors import RuntimeRequestError

DEFAULT_RUNTIME_SEARCH_LIMIT = DEFAULT_SEARCH_LIMIT
DEFAULT_RUNTIME_CONTEXT_LIMIT = DEFAULT_CONTEXT_LIMIT


@dataclass(frozen=True)
class IngestRuntimeRequest:
    path: str
    namespace: str
    corpus_id: str


@dataclass(frozen=True)
class RetrievalRuntimeRequest:
    query: str
    namespace: str
    corpus_ids: tuple[str, ...]
    limit: int
    content_types: tuple[str, ...] | None
    document_ids: tuple[str, ...] | None
    rerank: bool
    use_lexical_search: bool
    max_chars: int | None = None
    max_tokens: int | None = None


_INGEST_FIELDS = frozenset({"path", "namespace", "corpus_id"})
_RETRIEVAL_FIELDS = frozenset(
    {
        "query",
        "namespace",
        "corpus_ids",
        "limit",
        "content_types",
        "document_ids",
        "rerank",
        "use_lexical_search",
    }
)
_CONTEXT_RETRIEVAL_FIELDS = frozenset({*_RETRIEVAL_FIELDS, "max_chars", "max_tokens"})


def parse_ingest_request(payload: Mapping[str, object]) -> IngestRuntimeRequest:
    _reject_unknown_fields(payload, allowed=_INGEST_FIELDS)
    path = _text(payload, "path")
    namespace = _text(payload, "namespace")
    corpus_id = _text(payload, "corpus_id")
    missing = [
        field
        for field, value in (
            ("path", path),
            ("namespace", namespace),
            ("corpus_id", corpus_id),
        )
        if not value
    ]
    if missing:
        raise RuntimeRequestError(
            message="path, namespace, and corpus_id are required",
            details={"missing_fields": missing},
        )
    return IngestRuntimeRequest(path=path, namespace=namespace, corpus_id=corpus_id)


def parse_retrieval_request(
    payload: Mapping[str, object],
    *,
    default_limit: int = DEFAULT_RUNTIME_SEARCH_LIMIT,
    allow_context_budget: bool = False,
) -> RetrievalRuntimeRequest:
    _reject_unknown_fields(
        payload,
        allowed=_CONTEXT_RETRIEVAL_FIELDS if allow_context_budget else _RETRIEVAL_FIELDS,
    )
    query = _text(payload, "query")
    namespace = _text(payload, "namespace")
    corpus_id_list = _corpus_ids(payload, "corpus_ids")
    missing = [
        name
        for name, ok in (
            ("query", bool(query)),
            ("namespace", bool(namespace)),
            ("corpus_ids", bool(corpus_id_list)),
        )
        if not ok
    ]
    if missing:
        raise RuntimeRequestError(
            message="query, namespace, and corpus_ids are required",
            details={"missing_fields": missing},
        )
    return RetrievalRuntimeRequest(
        query=query,
        namespace=namespace,
        corpus_ids=tuple(corpus_id_list),
        limit=_limit(payload, default=default_limit),
        content_types=_optional_string_list(payload, "content_types"),
        document_ids=_optional_string_list(payload, "document_ids"),
        rerank=_bool(payload, "rerank", default=DEFAULT_RERANK),
        use_lexical_search=_bool(
            payload,
            "use_lexical_search",
            default=DEFAULT_USE_LEXICAL_SEARCH,
        ),
        max_chars=_optional_positive_int(payload, "max_chars")
        if allow_context_budget
        else None,
        max_tokens=_optional_positive_int(payload, "max_tokens")
        if allow_context_budget
        else None,
    )


def _reject_unknown_fields(
    payload: Mapping[str, object],
    *,
    allowed: frozenset[str],
) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise RuntimeRequestError(
            message="unexpected request fields",
            details={"fields": unknown},
        )


def _text(payload: Mapping[str, object], field: str) -> str:
    value = payload.get(field)
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    raise RuntimeRequestError(
        message=f"{field} must be a string",
        details={"field": field},
    )


def _corpus_ids(payload: Mapping[str, object], field: str) -> list[str]:
    value = payload.get(field)
    if value is None:
        return []
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise RuntimeRequestError(
            message=f"{field} must be an array of non-empty strings",
            details={"field": field},
        )
    return [item.strip() for item in value]


def _optional_string_list(
    payload: Mapping[str, object],
    field: str,
) -> tuple[str, ...] | None:
    if field not in payload:
        return None
    value = payload[field]
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise RuntimeRequestError(
            message=f"{field} must be an array of non-empty strings",
            details={"field": field},
        )
    return tuple(item.strip() for item in value)


def _limit(payload: Mapping[str, object], *, default: int) -> int:
    raw_limit = payload.get("limit")
    if raw_limit is None:
        return default
    if isinstance(raw_limit, bool) or not isinstance(raw_limit, int):
        raise RuntimeRequestError(
            message="limit must be an integer",
            details={"field": "limit"},
        )
    if raw_limit < 1:
        raise RuntimeRequestError(
            message="limit must be greater than zero",
            details={"field": "limit"},
        )
    return raw_limit


def _optional_positive_int(payload: Mapping[str, object], field: str) -> int | None:
    raw_value = payload.get(field)
    if raw_value is None:
        return None
    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        raise RuntimeRequestError(
            message=f"{field} must be an integer",
            details={"field": field},
        )
    if raw_value < 1:
        raise RuntimeRequestError(
            message=f"{field} must be greater than zero",
            details={"field": field},
        )
    return raw_value


def _bool(payload: Mapping[str, object], field: str, *, default: bool) -> bool:
    value = payload.get(field)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise RuntimeRequestError(
        message=f"{field} must be a boolean",
        details={"field": field},
    )
