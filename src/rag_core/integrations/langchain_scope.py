"""Shared scope normalization for optional LangChain adapters."""

from __future__ import annotations

from collections.abc import Sequence


def validate_langchain_namespace(namespace: str) -> str:
    normalized = namespace.strip()
    if not normalized:
        raise ValueError("namespace must not be empty")
    return normalized


def normalize_langchain_retrieval_scope(
    *,
    corpus_ids: Sequence[str],
    document_ids: Sequence[str] | None,
    limit: int,
) -> tuple[tuple[str, ...], tuple[str, ...] | None]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    corpus_ids_tuple = tuple(corpus_ids)
    if not corpus_ids_tuple:
        raise ValueError("corpus_ids must not be empty")
    document_ids_tuple = tuple(document_ids) if document_ids is not None else None
    return corpus_ids_tuple, document_ids_tuple


__all__ = ["normalize_langchain_retrieval_scope", "validate_langchain_namespace"]
