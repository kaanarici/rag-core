"""Conventions used by the engine to talk to a VectorStore.

Payload field names, point-ID format, document-ID format, and an optional
tenant-payload-index hint live here so adapters can override the shape and
multi-tenant indexing strategy without forcing edits inside the indexer or
filter helpers. The defaults preserve the Qdrant-shaped layout byte-for-byte.

``CollectionPolicy`` is an optional process fence: one ``Engine`` can bind to
a namespace and an allowed collection set so cross-scope requests fail at the
engine seam before provider egress.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Callable


class CollectionPolicyViolation(ValueError):
    """Raised when a request would exceed a bound ``CollectionPolicy``.

    Subclass of ``ValueError`` so existing seam-level guards (the request
    models' ``__post_init__`` checks, indexer namespace validation) catch
    it consistently as a contract violation.
    """


def _default_point_id(
    namespace: str,
    collection: str,
    document_id: str,
    chunk_index: int,
) -> str:
    raw = f"{namespace.strip()}::{collection.strip()}::{document_id}:chunk:{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def _default_document_id(
    namespace: str,
    collection: str,
    document_key: str,
) -> str:
    raw = f"{namespace.strip()}::{collection.strip()}::{document_key}"
    return f"doc_{uuid.uuid5(uuid.NAMESPACE_URL, raw)}"


@dataclass(frozen=True)
class VectorStorePolicy:
    """Field names, point/document ID formats, and tenant hint for a vector store."""

    namespace_field: str = "namespace"
    collection_field: str = "collection"
    document_id_field: str = "document_id"
    document_key_field: str = "document_key"
    content_sha256_field: str = "content_sha256"
    processing_version_field: str = "processing_version"
    content_type_field: str = "content_type"
    source_type_field: str = "source_type"
    chunk_index_field: str = "chunk_index"
    text_field: str = "text"
    title_field: str = "title"
    point_id_format: Callable[[str, str, str, int], str] = _default_point_id
    document_id_format: Callable[[str, str, str], str] = _default_document_id
    # When set, the Qdrant adapter creates the payload index for this field with
    # ``is_tenant=True`` (multi-tenant optimization). Other adapters ignore it.
    tenant_payload_field: str | None = None

    def make_point_id(
        self,
        *,
        namespace: str,
        collection: str,
        document_id: str,
        chunk_index: int,
    ) -> str:
        return self.point_id_format(namespace, collection, document_id, chunk_index)

    def make_document_id(
        self,
        *,
        namespace: str,
        collection: str,
        document_key: str,
    ) -> str:
        return self.document_id_format(namespace, collection, document_key)


DEFAULT_POLICY = VectorStorePolicy()


@dataclass(frozen=True)
class CollectionPolicy:
    """Per-process namespace and collection fence.

    All fields are optional; ``CollectionPolicy()`` is unrestricted.
    Violations always raise ``CollectionPolicyViolation``.
    """

    bound_namespace: str | None = None
    allowed_collections: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if self.bound_namespace is not None and (
            not isinstance(self.bound_namespace, str)
            or not self.bound_namespace.strip()
        ):
            raise ValueError(
                "CollectionPolicy.bound_namespace must be None or a non-empty string"
            )
        if self.allowed_collections is not None:
            if not isinstance(self.allowed_collections, frozenset):
                raise ValueError(
                    "CollectionPolicy.allowed_collections must be a frozenset[str]"
                )
            for value in self.allowed_collections:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        "CollectionPolicy.allowed_collections must contain non-empty strings"
                    )

    def _emit(self, message: str) -> None:
        raise CollectionPolicyViolation(message)

    def validate_namespace(self, namespace: str) -> None:
        if self.bound_namespace is None:
            return
        if namespace != self.bound_namespace:
            self._emit(
                f"CollectionPolicy bound to namespace={self.bound_namespace!r}; "
                f"refused request for namespace={namespace!r}"
            )

    def validate_collections(self, collections: list[str] | None) -> None:
        if self.allowed_collections is None:
            return
        if collections is None:
            self._emit(
                "CollectionPolicy.allowed_collections is set; request must pass an "
                "explicit non-empty collections list (None silently widens)"
            )
            return
        for collection in collections:
            if collection not in self.allowed_collections:
                self._emit(
                    f"CollectionPolicy refused collection={collection!r}; "
                    f"allowed={sorted(self.allowed_collections)!r}"
                )

    def validate_search(
        self,
        *,
        namespace: str,
        collections: list[str] | None,
    ) -> None:
        self.validate_namespace(namespace)
        self.validate_collections(collections)

    def validate_delete(
        self,
        *,
        namespace: str,
        collection: str | None,
    ) -> None:
        self.validate_namespace(namespace)
        if collection is not None:
            self.validate_collections([collection])


DEFAULT_COLLECTION_POLICY = CollectionPolicy()
