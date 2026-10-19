"""Conventions used by the engine to talk to a VectorStore.

Payload field names, point-ID format, document-ID format, and an optional
tenant-payload-index hint live here so adapters can override the shape and
multi-tenant indexing strategy without forcing edits inside the indexer or
filter helpers. Default IDs encode each scope component separately.

``CollectionPolicy`` is an optional process fence: one ``Engine`` can bind to
a tenant_id and an allowed collection set so cross-scope requests fail at the
engine seam before provider egress.
"""

from __future__ import annotations

import uuid
import json
from dataclasses import dataclass
from typing import Callable

from rag_core.search.stored_payload_fields import FIXED_PAYLOAD_KEYS


class CollectionPolicyViolation(ValueError):
    """Raised when a request would exceed a bound ``CollectionPolicy``.

    Subclass of ``ValueError`` so existing seam-level guards (the request
    models' ``__post_init__`` checks, indexer tenant_id validation) catch
    it consistently as a contract violation.
    """


def scoped_uuid(kind: str, *parts: str | int) -> str:
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            json.dumps([kind, *parts], ensure_ascii=False, separators=(",", ":")),
        )
    )


def _default_point_id(
    tenant_id: str, collection: str, document_id: str, chunk_index: int
) -> str:
    return scoped_uuid(
        "point-v2", tenant_id.strip(), collection.strip(), document_id, chunk_index
    )


def _default_document_id(tenant_id: str, collection: str, document_key: str) -> str:
    return "doc_v2_" + scoped_uuid(
        "document-v2", tenant_id.strip(), collection.strip(), document_key
    )


@dataclass(frozen=True)
class VectorStorePolicy:
    """Field names, point/document ID formats, and tenant hint for a vector store."""

    tenant_id_field: str = "tenant_id"
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

    def __post_init__(self) -> None:
        fields = [
            getattr(self, name)
            for name in (
                "tenant_id_field",
                "collection_field",
                "document_id_field",
                "document_key_field",
                "content_sha256_field",
                "processing_version_field",
                "content_type_field",
                "source_type_field",
                "chunk_index_field",
                "text_field",
                "title_field",
            )
        ]
        if any(
            not isinstance(name, str)
            or not name.strip()
            or name != name.strip()
            or name.startswith("_rag_")
            or name in FIXED_PAYLOAD_KEYS
            for name in fields
        ) or len(set(fields)) != len(fields):
            raise ValueError(
                "VectorStorePolicy fields must be distinct non-empty names outside reserved document metadata"
            )

    def make_point_id(
        self,
        *,
        tenant_id: str,
        collection: str,
        document_id: str,
        chunk_index: int,
    ) -> str:
        return self.point_id_format(tenant_id, collection, document_id, chunk_index)

    def make_document_id(
        self,
        *,
        tenant_id: str,
        collection: str,
        document_key: str,
    ) -> str:
        return self.document_id_format(tenant_id, collection, document_key)


DEFAULT_POLICY = VectorStorePolicy()


@dataclass(frozen=True)
class CollectionPolicy:
    """Per-process tenant_id and collection fence.

    All fields are optional; ``CollectionPolicy()`` is unrestricted.
    Violations always raise ``CollectionPolicyViolation``.
    """

    bound_tenant_id: str | None = None
    allowed_collections: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if self.bound_tenant_id is not None and (
            not isinstance(self.bound_tenant_id, str)
            or not self.bound_tenant_id.strip()
        ):
            raise ValueError(
                "CollectionPolicy.bound_tenant_id must be None or a non-empty string"
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

    def validate_tenant_id(self, tenant_id: str) -> None:
        if self.bound_tenant_id is None:
            return
        if tenant_id != self.bound_tenant_id:
            self._emit(
                f"CollectionPolicy bound to tenant_id={self.bound_tenant_id!r}; "
                f"refused request for tenant_id={tenant_id!r}"
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
        tenant_id: str,
        collections: list[str] | None,
    ) -> None:
        self.validate_tenant_id(tenant_id)
        self.validate_collections(collections)

    def validate_ingest(self, *, tenant_id: str, collection: str) -> None:
        self.validate_tenant_id(tenant_id)
        self.validate_collections([collection])

    def validate_delete(
        self,
        *,
        tenant_id: str,
        collection: str | None,
    ) -> None:
        self.validate_tenant_id(tenant_id)
        if collection is not None:
            self.validate_collections([collection])


DEFAULT_COLLECTION_POLICY = CollectionPolicy()


def validate_ingest_scope(
    tenant_id: str | None, collection: str | None, policy: CollectionPolicy | None
) -> tuple[str, str]:
    from rag_core.scope import normalize_tenant_id, normalize_collection

    tenant, name = normalize_tenant_id(tenant_id), normalize_collection(collection)
    if policy is not None:
        policy.validate_ingest(tenant_id=tenant, collection=name)
    return tenant, name
