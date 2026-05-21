"""Conventions used by the engine to talk to a VectorStore.

Payload field names, point-ID format, document-ID format, and an optional
tenant-payload-index hint live here so adapters can override the shape and
multi-tenant indexing strategy without forcing edits inside the indexer or
filter helpers. The defaults preserve the Qdrant-shaped layout byte-for-byte.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Callable


def _default_point_id(
    namespace: str,
    corpus_id: str,
    document_id: str,
    chunk_index: int,
) -> str:
    raw = f"{namespace.strip()}::{corpus_id.strip()}::{document_id}:chunk:{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def _default_document_id(
    namespace: str,
    corpus_id: str,
    document_key: str,
) -> str:
    raw = f"{namespace.strip()}::{corpus_id.strip()}::{document_key}"
    return f"doc_{uuid.uuid5(uuid.NAMESPACE_URL, raw)}"


@dataclass(frozen=True)
class VectorStorePolicy:
    """Field names, point/document ID formats, and tenant hint for a vector store."""

    namespace_field: str = "namespace"
    corpus_id_field: str = "corpus_id"
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
        corpus_id: str,
        document_id: str,
        chunk_index: int,
    ) -> str:
        return self.point_id_format(namespace, corpus_id, document_id, chunk_index)

    def make_document_id(
        self,
        *,
        namespace: str,
        corpus_id: str,
        document_key: str,
    ) -> str:
        return self.document_id_format(namespace, corpus_id, document_key)


DEFAULT_POLICY = VectorStorePolicy()
