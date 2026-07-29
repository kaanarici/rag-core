"""Typed request parsing for the optional HTTP runtime."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass

from rag_core.contracts.tool_contract_requests import (
    normalize_static_retrieval_scope,
    validate_bound_namespace,
    validate_limit_bounds,
)
from rag_core.contracts.tool_contract_schemas import (
    SEARCH_USER_DOCUMENTS_LIMIT_MAX,
    SEARCH_USER_DOCUMENTS_LIMIT_MIN,
    SEARCH_USER_DOCUMENTS_MAX_CHARS_MAX,
    SEARCH_USER_DOCUMENTS_MAX_CHARS_MIN,
    SEARCH_USER_DOCUMENTS_MAX_TOKENS_MAX,
    SEARCH_USER_DOCUMENTS_MAX_TOKENS_MIN,
)
from rag_core.retrieval_defaults import (
    DEFAULT_CONTEXT_LIMIT,
    DEFAULT_RERANK,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_USE_LEXICAL_SEARCH,
)
from rag_core.search.context_pack import ContextOrder, validate_context_order
from rag_core.runtime.errors import RuntimeRequestError
from rag_core.scope import DEFAULT_NAMESPACE

DEFAULT_RUNTIME_SEARCH_LIMIT = DEFAULT_SEARCH_LIMIT
DEFAULT_RUNTIME_CONTEXT_LIMIT = DEFAULT_CONTEXT_LIMIT


@dataclass(frozen=True)
class IngestRuntimeRequest:
    path: str
    namespace: str
    collection: str


@dataclass(frozen=True)
class DeleteDocumentRuntimeRequest:
    """Parsed body for ``DELETE /v1/documents/{document_id}``.

    The ``document_id`` comes from the URL path; ``namespace`` and
    ``collection`` are body-or-query parameters because right-to-forget must
    fail closed if the caller doesn't claim a scope explicitly.
    """

    document_id: str
    namespace: str
    collection: str


@dataclass(frozen=True)
class RetrievalRuntimeRequest:
    query: str
    namespace: str
    collections: tuple[str, ...]
    limit: int
    content_types: tuple[str, ...] | None
    document_ids: tuple[str, ...] | None
    rerank: bool
    use_lexical_search: bool
    max_chars: int | None = None
    max_tokens: int | None = None
    context_order: ContextOrder = "rank"


_INGEST_FIELDS = frozenset({"path", "namespace", "collection", "corpus_id"})
_DELETE_DOCUMENT_FIELDS = frozenset({"namespace", "collection", "corpus_id"})
_RETRIEVAL_FIELDS = frozenset(
    {
        "query",
        "namespace",
        "collection",
        "collections",
        "corpus_id",
        "corpus_ids",
        "limit",
        "content_types",
        "document_ids",
        "rerank",
        "use_lexical_search",
    }
)
_CONTEXT_RETRIEVAL_FIELDS = frozenset(
    {*_RETRIEVAL_FIELDS, "max_chars", "max_tokens", "context_order"}
)


def parse_delete_document_request(
    *,
    document_id: str,
    payload: Mapping[str, object],
    bound_namespace: str | None = None,
) -> DeleteDocumentRuntimeRequest:
    """Parse the body for ``DELETE /v1/documents/{document_id}``.

    Mirrors :func:`parse_ingest_request`. ``document_id`` is the URL path
    parameter; ``namespace`` and ``collection`` are required body fields so
    right-to-forget fails closed if the caller forgets the tier scope.

    When ``bound_namespace`` is set for a one-namespace deployment, the
    caller cannot override the namespace from the body. The body field is
    enforced equal via :func:`validate_bound_namespace`.
    """
    _reject_unknown_fields(payload, allowed=_DELETE_DOCUMENT_FIELDS)
    document_id_value = (document_id or "").strip()
    if not document_id_value:
        raise RuntimeRequestError(
            message="document_id is required in the URL path",
            details={"missing_fields": ["document_id"]},
        )
    namespace = _namespace_from_payload(payload, bound_namespace=bound_namespace)
    collection = _collection_from_payload(payload)
    missing = [
        field
        for field, value in (
            ("collection", collection),
        )
        if not value
    ]
    if missing:
        raise RuntimeRequestError(
            message="collection is required",
            details={"missing_fields": missing},
        )
    return DeleteDocumentRuntimeRequest(
        document_id=document_id_value,
        namespace=namespace,
        collection=collection,
    )


def parse_ingest_request(
    payload: Mapping[str, object],
    *,
    bound_namespace: str | None = None,
) -> IngestRuntimeRequest:
    _reject_unknown_fields(payload, allowed=_INGEST_FIELDS)
    path = _text(payload, "path")
    namespace = _namespace_from_payload(payload, bound_namespace=bound_namespace)
    collection = _collection_from_payload(payload)
    missing = [
        field
        for field, value in (
            ("path", path),
            ("collection", collection),
        )
        if not value
    ]
    if missing:
        raise RuntimeRequestError(
            message="path and collection are required",
            details={"missing_fields": missing},
        )
    return IngestRuntimeRequest(path=path, namespace=namespace, collection=collection)


def parse_retrieval_request(
    payload: Mapping[str, object],
    *,
    default_limit: int = DEFAULT_RUNTIME_SEARCH_LIMIT,
    allow_context_budget: bool = False,
    bound_namespace: str | None = None,
) -> RetrievalRuntimeRequest:
    _reject_unknown_fields(
        payload,
        allowed=_CONTEXT_RETRIEVAL_FIELDS if allow_context_budget else _RETRIEVAL_FIELDS,
    )
    query = _text(payload, "query")
    namespace = _namespace_from_payload(payload, bound_namespace=bound_namespace)
    collection_list = _collections_from_payload(payload)
    # Distinguish ABSENT from PRESENT-BUT-INVALID. An empty list belongs in
    # the contract normalizer branch below, which is the single seam every
    # in-process caller flows through; the missing_fields branch is reserved
    # for fields entirely omitted from the request body.
    has_scope_field = any(
        field in payload
        for field in ("collection", "collections", "corpus_id", "corpus_ids")
    )
    missing = [
        name
        for name, ok in (("query", bool(query)), ("collection", has_scope_field))
        if not ok
    ]
    if missing:
        raise RuntimeRequestError(
            message="query and collection are required",
            details={"missing_fields": missing},
        )
    limit_value = _limit(payload, default=default_limit)
    document_ids_value = _optional_string_list(payload, "document_ids")
    # Run the contract normalizer so collections / document_ids are validated
    # by the same code path the in-process callers use. This rejects empty
    # collections at the HTTP boundary before the engine sees the request.
    try:
        collections_tuple, document_ids_tuple = normalize_static_retrieval_scope(
            collections=tuple(collection_list),
            document_ids=document_ids_value,
            limit=limit_value,
        )
    except ValueError as exc:
        raise RuntimeRequestError(
            message=str(exc),
            details={"field": "collections"},
        ) from None
    return RetrievalRuntimeRequest(
        query=query,
        namespace=namespace,
        collections=collections_tuple,
        limit=limit_value,
        content_types=_optional_string_list(payload, "content_types"),
        document_ids=document_ids_tuple,
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
        context_order=_context_order(payload) if allow_context_budget else "rank",
    )


def _namespace_from_payload(
    payload: Mapping[str, object],
    *,
    bound_namespace: str | None,
) -> str:
    namespace = _text(payload, "namespace") if "namespace" in payload else ""
    if not namespace and bound_namespace is not None:
        return bound_namespace
    if not namespace:
        return DEFAULT_NAMESPACE
    return _enforce_bound_namespace(namespace, bound_namespace=bound_namespace)


def _enforce_bound_namespace(
    namespace: str,
    *,
    bound_namespace: str | None,
) -> str:
    """Normalize ``namespace`` and enforce the process-bound tenant.

    Routes through :func:`validate_bound_namespace` so the runtime uses the
    same validator the in-process integrations use. When ``bound_namespace``
    is non-None for a single-namespace process, the body field must
    match exactly. The gateway cannot mint a request that retrieves from a
    different workspace just by editing the body.
    """
    try:
        normalized = validate_bound_namespace(namespace)
    except ValueError as exc:
        raise RuntimeRequestError(
            message=str(exc),
            details={"field": "namespace"},
        ) from None
    if bound_namespace is not None and normalized != bound_namespace:
        raise RuntimeRequestError(
            message="namespace does not match process-bound tenant",
            details={"field": "namespace"},
        )
    return normalized


def _collection_from_payload(payload: Mapping[str, object]) -> str:
    if "collection" in payload and "corpus_id" in payload:
        raise RuntimeRequestError(
            message="collection and corpus_id are mutually exclusive",
            details={"fields": ["collection", "corpus_id"]},
        )
    if "corpus_id" in payload:
        _warn_deprecated_field("corpus_id", "collection")
        return _text(payload, "corpus_id")
    return _text(payload, "collection")


def _collections_from_payload(payload: Mapping[str, object]) -> list[str]:
    present = [
        field
        for field in ("collection", "collections", "corpus_id", "corpus_ids")
        if field in payload
    ]
    if len(present) > 1:
        raise RuntimeRequestError(
            message="collection fields are mutually exclusive",
            details={"fields": present},
        )
    if not present:
        return []
    field = present[0]
    if field == "collection":
        collection = _text(payload, "collection")
        return [collection] if collection else []
    if field == "corpus_id":
        _warn_deprecated_field("corpus_id", "collection")
        collection = _text(payload, "corpus_id")
        return [collection] if collection else []
    if field == "corpus_ids":
        _warn_deprecated_field("corpus_ids", "collections")
    return _collections(payload, field)


def _warn_deprecated_field(old: str, new: str) -> None:
    warnings.warn(
        f"HTTP request field {old!r} is deprecated; use {new!r}",
        DeprecationWarning,
        stacklevel=3,
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


def _collections(payload: Mapping[str, object], field: str) -> list[str]:
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
    try:
        validate_limit_bounds(
            raw_limit,
            minimum=SEARCH_USER_DOCUMENTS_LIMIT_MIN,
            maximum=SEARCH_USER_DOCUMENTS_LIMIT_MAX,
            field="limit",
        )
    except ValueError as exc:
        raise RuntimeRequestError(
            message=str(exc),
            details={"field": "limit"},
        ) from None
    return raw_limit


_RUNTIME_OPTIONAL_BOUNDS: dict[str, tuple[int, int]] = {
    "max_chars": (SEARCH_USER_DOCUMENTS_MAX_CHARS_MIN, SEARCH_USER_DOCUMENTS_MAX_CHARS_MAX),
    "max_tokens": (
        SEARCH_USER_DOCUMENTS_MAX_TOKENS_MIN,
        SEARCH_USER_DOCUMENTS_MAX_TOKENS_MAX,
    ),
}


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
    bounds = _RUNTIME_OPTIONAL_BOUNDS.get(field)
    if bounds is not None:
        minimum, maximum = bounds
        try:
            validate_limit_bounds(
                raw_value,
                minimum=minimum,
                maximum=maximum,
                field=field,
            )
        except ValueError as exc:
            raise RuntimeRequestError(
                message=str(exc),
                details={"field": field},
            ) from None
    return raw_value


def _context_order(payload: Mapping[str, object]) -> ContextOrder:
    raw_value = payload.get("context_order")
    if raw_value is None:
        return "rank"
    if not isinstance(raw_value, str):
        raise RuntimeRequestError(
            message="context_order must be a string",
            details={"field": "context_order"},
        )
    try:
        return validate_context_order(raw_value)
    except ValueError as exc:
        raise RuntimeRequestError(
            message=str(exc),
            details={"field": "context_order"},
        ) from None


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
