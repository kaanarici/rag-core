"""Conventions used by the engine to talk to a VectorStore.

Payload field names, point-ID format, document-ID format, and an optional
tenant-payload-index hint live here so adapters can override the shape and
multi-tenant indexing strategy without forcing edits inside the indexer or
filter helpers. The defaults preserve the Qdrant-shaped layout byte-for-byte.

``CorpusPolicy`` is a separate, optional seam: a single ``RAGCore`` process
can be bound to one namespace, an allowed corpus-id set, and a capability
subset. That lets a sensitive tier refuse rerank, lexical sidecar, cache, or
non-allowed query-plan usage at the engine seam before provider egress.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Callable


# Per-tier collection isolation. The slug is folded into the Qdrant collection
# name so independently configured corpora point at physically distinct
# collections.
_CORPUS_SLUG_SAFE = re.compile(r"[^a-z0-9_-]+")


def corpus_slug_for(corpus_id: str) -> str:
    """Deterministic ``[a-z0-9_-]`` slug used in collection naming."""

    if not isinstance(corpus_id, str) or not corpus_id.strip():
        raise ValueError("corpus_id must be a non-empty string to slug")
    lowered = corpus_id.strip().lower()
    sanitized = _CORPUS_SLUG_SAFE.sub("_", lowered)
    collapsed = re.sub(r"_+", "_", sanitized).strip("_")
    if not collapsed:
        raise ValueError(
            f"corpus_id={corpus_id!r} produces an empty slug after sanitization"
        )
    return collapsed


class CorpusPolicyViolation(ValueError):
    """Raised when a request would exceed a bound ``CorpusPolicy``.

    Subclass of ``ValueError`` so existing seam-level guards (the request
    models' ``__post_init__`` checks, indexer namespace validation) catch
    it consistently as a contract violation.
    """


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


@dataclass(frozen=True)
class CorpusPolicy:
    """Per-process retrieval/delete capability fence.

    All fields are optional; ``CorpusPolicy()`` (all defaults) is a no-op
    seam useful only to assert "policy was wired but unrestricted." A
    restricted-tier instance typically sets ``bound_namespace``,
    ``allowed_corpus_ids``, and turns ``allow_rerank`` /
    ``allow_lexical_sidecar`` off.

    Violations always raise ``CorpusPolicyViolation``. A tenancy/capability
    fence must not degrade to a warning.
    """

    bound_namespace: str | None = None
    allowed_corpus_ids: frozenset[str] | None = None
    allow_rerank: bool = True
    allow_lexical_sidecar: bool = True
    allow_sparse: bool = True
    allowed_query_plan_presets: frozenset[str] | None = None
    # Sensitive-tier deploy switch: when True, ``core_assembly`` swaps the
    # configured embedding and chunk-context caches for no-op providers.
    cache_disabled: bool = False

    def __post_init__(self) -> None:
        if self.bound_namespace is not None and (
            not isinstance(self.bound_namespace, str)
            or not self.bound_namespace.strip()
        ):
            raise ValueError(
                "CorpusPolicy.bound_namespace must be None or a non-empty string"
            )
        if self.allowed_corpus_ids is not None:
            if not isinstance(self.allowed_corpus_ids, frozenset):
                raise ValueError(
                    "CorpusPolicy.allowed_corpus_ids must be a frozenset[str]"
                )
            for value in self.allowed_corpus_ids:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        "CorpusPolicy.allowed_corpus_ids must contain non-empty strings"
                    )
        if self.allowed_query_plan_presets is not None:
            if not isinstance(self.allowed_query_plan_presets, frozenset):
                raise ValueError(
                    "CorpusPolicy.allowed_query_plan_presets must be a frozenset[str]"
                )
            for value in self.allowed_query_plan_presets:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        "CorpusPolicy.allowed_query_plan_presets must contain non-empty strings"
                    )
    def _emit(self, message: str) -> None:
        raise CorpusPolicyViolation(message)

    def validate_namespace(self, namespace: str) -> None:
        if self.bound_namespace is None:
            return
        if namespace != self.bound_namespace:
            self._emit(
                f"CorpusPolicy bound to namespace={self.bound_namespace!r}; "
                f"refused request for namespace={namespace!r}"
            )

    def validate_corpus_ids(self, corpus_ids: list[str] | None) -> None:
        if self.allowed_corpus_ids is None:
            return
        if corpus_ids is None:
            self._emit(
                "CorpusPolicy.allowed_corpus_ids is set; request must pass an "
                "explicit non-empty corpus_ids list (None silently widens)"
            )
            return
        for corpus_id in corpus_ids:
            if corpus_id not in self.allowed_corpus_ids:
                self._emit(
                    f"CorpusPolicy refused corpus_id={corpus_id!r}; "
                    f"allowed={sorted(self.allowed_corpus_ids)!r}"
                )

    def validate_search(
        self,
        *,
        namespace: str,
        corpus_ids: list[str] | None,
        rerank: bool,
        use_lexical_search: bool,
        query_plan_preset: str | None = None,
    ) -> None:
        self.validate_namespace(namespace)
        self.validate_corpus_ids(corpus_ids)
        if rerank and not self.allow_rerank:
            self._emit("CorpusPolicy disallows rerank on this tier")
        if use_lexical_search and not self.allow_lexical_sidecar:
            self._emit("CorpusPolicy disallows the lexical sidecar on this tier")
        if (
            self.allowed_query_plan_presets is not None
            and query_plan_preset is not None
            and query_plan_preset not in self.allowed_query_plan_presets
        ):
            self._emit(
                f"CorpusPolicy refused query_plan_preset={query_plan_preset!r}; "
                f"allowed={sorted(self.allowed_query_plan_presets)!r}"
            )

    def validate_delete(
        self,
        *,
        namespace: str,
        corpus_id: str | None,
    ) -> None:
        self.validate_namespace(namespace)
        if corpus_id is not None:
            self.validate_corpus_ids([corpus_id])

    @property
    def collection_corpus_slug(self) -> str | None:
        """Return a slug iff this policy binds the process to one corpus tier.

        When ``allowed_corpus_ids`` is a single-element set, the slug becomes
        part of the Qdrant collection name so sensitive and public corpora do
        not share a physical collection. Multi-corpus processes intentionally
        return ``None`` so the legacy single-collection layout still works.
        """

        if self.allowed_corpus_ids is None or len(self.allowed_corpus_ids) != 1:
            return None
        (only,) = self.allowed_corpus_ids
        return corpus_slug_for(only)


DEFAULT_CORPUS_POLICY = CorpusPolicy()
