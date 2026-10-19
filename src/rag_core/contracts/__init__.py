"""App-bound retrieval scope, shared by the facade and MCP."""

from __future__ import annotations
from collections.abc import Sequence
from rag_core.scope import resolve_collections_argument
from rag_core.search.request_models import _require_positive_int


def validate_bound_tenant_id(tenant_id: str) -> str:
    """Normalize the app-bound tenant_id before retrieval."""
    normalized = tenant_id.strip()
    if not normalized:
        raise ValueError("tenant_id must not be empty")
    return normalized


def normalize_static_retrieval_scope(
    *,
    collection: str | None = None,
    collections: Sequence[str] | None = None,
    document_ids: Sequence[str] | None,
    limit: int,
) -> tuple[tuple[str, ...], tuple[str, ...] | None]:
    """Normalize app-bound collection and document scope."""
    _require_positive_int(limit, "limit")
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
    rejected = [
        document_id for document_id in requested if document_id not in configured_set
    ]
    if rejected:
        raise ValueError(
            "document_ids contain values outside the configured retrieval scope"
        )
    return list(requested)


def _normalize_scope_values(values: Sequence[str], field: str) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must contain non-empty strings")
        normalized.append(value.strip())
    return tuple(normalized)
