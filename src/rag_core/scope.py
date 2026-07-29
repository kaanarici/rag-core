from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_NAMESPACE = "default"


@dataclass(frozen=True)
class Scope:
    """Tenant and collection boundary for ingest, search, and delete."""

    tenant_id: str
    collection: str
    document_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", normalize_namespace(self.tenant_id))
        object.__setattr__(self, "collection", normalize_collection(self.collection))
        if self.document_ids is None:
            return
        object.__setattr__(
            self,
            "document_ids",
            tuple(
                _normalize_string_sequence(
                    self.document_ids,
                    field="document_ids",
                )
            ),
        )

    @property
    def namespace(self) -> str:
        """Internal storage name for ``tenant_id``."""

        return self.tenant_id


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
    resolved = _normalize_string_sequence(collections, field="collections")
    if not resolved:
        raise ValueError("collections must not be empty")
    return resolved


def _normalize_string_sequence(
    values: Sequence[str],
    *,
    field: str,
) -> list[str]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise ValueError(f"{field} must be a sequence of strings")
    resolved: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must contain non-empty strings")
        resolved.append(value.strip())
    return resolved


__all__ = [
    "DEFAULT_NAMESPACE",
    "Scope",
    "normalize_collection",
    "normalize_namespace",
    "resolve_collections_argument",
]
