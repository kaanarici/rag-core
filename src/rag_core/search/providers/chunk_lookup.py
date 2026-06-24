"""Shared validation for chunk-index lookup helpers."""

from __future__ import annotations

from collections.abc import Sequence


def validate_chunk_lookup_inputs(
    *,
    namespace: str,
    collection: str,
    document_id: str,
    chunk_indices: Sequence[int],
) -> tuple[str, str, str, tuple[int, ...]]:
    namespace_scoped = namespace.strip()
    if not namespace_scoped:
        raise ValueError("namespace is required for get_chunks_by_index")
    collection_scoped = collection.strip()
    if not collection_scoped:
        raise ValueError("collection is required for get_chunks_by_index")
    document_scoped = document_id.strip()
    if not document_scoped:
        raise ValueError("document_id is required for get_chunks_by_index")
    indices: list[int] = []
    seen: set[int] = set()
    for value in chunk_indices:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("chunk_indices must contain non-negative integers")
        if value in seen:
            continue
        seen.add(value)
        indices.append(value)
    return namespace_scoped, collection_scoped, document_scoped, tuple(indices)


__all__ = ["validate_chunk_lookup_inputs"]
