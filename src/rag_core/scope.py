from __future__ import annotations

from collections.abc import Sequence

DEFAULT_NAMESPACE = "default"


def normalize_namespace(namespace: str | None) -> str:
    value = DEFAULT_NAMESPACE if namespace is None else namespace
    if not isinstance(value, str) or not value.strip():
        raise ValueError("namespace must not be empty")
    return value.strip()


def normalize_collection(collection: str | None, *, field: str = "collection") -> str:
    if not isinstance(collection, str) or not collection.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return collection.strip()


def resolve_collections_argument(
    *,
    collection: str | None,
    collections: Sequence[str] | None,
    caller: str,
) -> list[str]:
    if collection is not None and collections is not None:
        raise TypeError(f"{caller} got both collection and collections")
    if collection is not None:
        return [normalize_collection(collection)]
    if collections is None:
        raise ValueError(f"{caller} requires collection or collections")
    resolved = _normalize_collection_sequence(collections)
    if not resolved:
        raise ValueError("collections must not be empty")
    return resolved


def _normalize_collection_sequence(values: Sequence[str]) -> list[str]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise ValueError("collections must be a sequence of strings")
    resolved: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("collections must contain non-empty strings")
        resolved.append(value.strip())
    return resolved


__all__ = [
    "DEFAULT_NAMESPACE",
    "normalize_collection",
    "normalize_namespace",
    "resolve_collections_argument",
]
