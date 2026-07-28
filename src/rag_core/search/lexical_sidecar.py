"""Portable lexical sidecar for exact or trigram-style matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

from rag_core.events.emit import emit_event
from rag_core.events.search_events import LexicalSidecarBoundExceeded
from rag_core.search.lexical_sidecar_matching import (
    LexicalMatch,
    match_lexical_result,
    normalized_lexical_query,
)
from rag_core.search.providers.registry import SEARCH_SIDECARS
from rag_core.search.provider_protocols import SearchSidecar
from rag_core.search.request_models import SearchSidecarQuery
from rag_core.search.vector_models import SearchResult

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink

# Bound defaults for the in-memory lexical sidecar. Sized for the
# small workspace (~5 analyst users, ~10s of MB of indexed text);
# anything bigger should switch to a real lexical store (BM25/SPLADE on disk)
# rather than holding it in this process's heap.
DEFAULT_LEXICAL_SIDECAR_MAX_ENTRIES: int = 100_000
DEFAULT_LEXICAL_SIDECAR_MAX_BYTES: int = 256 * 1024 * 1024


@dataclass(frozen=True)
class LexicalSidecarRecord:
    """Portable record shape for sidecar lookups."""

    namespace: str
    result: SearchResult


class PortableLexicalSidecar(SearchSidecar):
    """Small in-memory sidecar for exact and trigram-style retrieval.

    Bounded by ``max_entries`` and ``max_bytes`` so a runaway ingest cannot
    silently exhaust the process heap. When an upsert would push either
    bound over the limit, the sidecar emits ``LexicalSidecarBoundExceeded``
    (if an ``event_sink`` is wired) and refuses the overflow rather than
    OOMing.
    """

    provider_name = "portable_lexical"

    def __init__(
        self,
        records: list[LexicalSidecarRecord],
        *,
        trigram_threshold: float = 0.35,
        max_entries: int = DEFAULT_LEXICAL_SIDECAR_MAX_ENTRIES,
        max_bytes: int = DEFAULT_LEXICAL_SIDECAR_MAX_BYTES,
        event_sink: "EventSink | None" = None,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self._records = list(records)
        self._trigram_threshold = trigram_threshold
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._event_sink = event_sink
        self._current_bytes = sum(
            _lexical_record_bytes(record) for record in self._records
        )

    def upsert_records(self, records: Sequence[object]) -> None:
        existing = {
            (record.namespace, record.result.id): record
            for record in self._records
        }
        current_bytes = self._current_bytes
        rejected = 0
        bound_reason: str | None = None
        for record in records:
            if not isinstance(record, LexicalSidecarRecord):
                continue
            key = (record.namespace, record.result.id)
            incoming_bytes = _lexical_record_bytes(record)
            if key in existing:
                # Update in place. Only the byte delta matters; entry count
                # does not change.
                prior_bytes = _lexical_record_bytes(existing[key])
                projected_bytes = current_bytes - prior_bytes + incoming_bytes
                if projected_bytes > self._max_bytes:
                    rejected += 1
                    bound_reason = bound_reason or "max_bytes"
                    continue
                existing[key] = record
                current_bytes = projected_bytes
                continue
            # New entry. Needs to fit both bounds.
            if len(existing) + 1 > self._max_entries:
                rejected += 1
                bound_reason = bound_reason or "max_entries"
                continue
            projected_bytes = current_bytes + incoming_bytes
            if projected_bytes > self._max_bytes:
                rejected += 1
                bound_reason = bound_reason or "max_bytes"
                continue
            existing[key] = record
            current_bytes = projected_bytes
        self._records = list(existing.values())
        self._current_bytes = current_bytes
        if rejected and bound_reason is not None:
            emit_event(
                self._event_sink,
                LexicalSidecarBoundExceeded(
                    provider=self.provider_name,
                    reason=bound_reason,
                    rejected_count=rejected,
                    current_entries=len(self._records),
                    max_entries=self._max_entries,
                    current_bytes=self._current_bytes,
                    max_bytes=self._max_bytes,
                ),
            )

    def delete_document(
        self,
        *,
        namespace: str,
        document_id: str,
        collection: str | None = None,
    ) -> None:
        kept: list[LexicalSidecarRecord] = []
        kept_bytes = 0
        for record in self._records:
            matches_namespace = record.namespace == namespace
            matches_document = record.result.document_id == document_id
            matches_collection = collection is None or record.result.collection == collection
            if matches_namespace and matches_document and matches_collection:
                continue
            kept.append(record)
            kept_bytes += _lexical_record_bytes(record)
        self._records = kept
        self._current_bytes = kept_bytes

    @property
    def entry_count(self) -> int:
        return len(self._records)

    @property
    def byte_usage(self) -> int:
        return self._current_bytes

    async def search(self, query: SearchSidecarQuery) -> list[SearchResult]:
        needle = normalized_lexical_query(query.query)
        if not needle:
            return []

        matches: list[LexicalMatch] = []
        for record in self._records:
            match = match_lexical_result(
                record_namespace=record.namespace,
                result=record.result,
                query=query,
                needle=needle,
                trigram_threshold=self._trigram_threshold,
            )
            if match is not None:
                matches.append(match)

        matches.sort(key=lambda item: (item[0], item[1].score), reverse=True)
        return [result for _, result in matches[: query.limit]]


def _build_portable_lexical_sidecar(**kwargs: Any) -> PortableLexicalSidecar:
    records = kwargs.pop("records", None) or []
    return PortableLexicalSidecar(list(records), **kwargs)


def _lexical_record_bytes(record: LexicalSidecarRecord) -> int:
    # Approximate the heap cost of a sidecar record by the byte length of its
    # text + metadata text content. This is intentionally conservative. The
    # bound is a guardrail against runaway ingest, not a precise heap sizer.
    result = record.result
    total = 0
    text = getattr(result, "text", None)
    if isinstance(text, str):
        total += len(text.encode("utf-8"))
    embedding_text = getattr(result, "embedding_text", None)
    if isinstance(embedding_text, str) and embedding_text is not text:
        total += len(embedding_text.encode("utf-8"))
    title = getattr(result, "title", None)
    if isinstance(title, str):
        total += len(title.encode("utf-8"))
    # Add a small fixed overhead for the record + dict slot bookkeeping.
    return total + 256


def create_search_sidecar(
    provider: str | None = None,
    **kwargs: Any,
) -> SearchSidecar | None:
    """Resolve the SearchSidecar provider category from a config name.

    ``None`` means no sidecar is wired in, matching the default
    ``IngestConfig.enable_lexical_search=False`` behavior.
    """
    if provider is None:
        return None
    return SEARCH_SIDECARS.create(provider, **kwargs)


PORTABLE_LEXICAL_SIDECAR_PROVIDER = PortableLexicalSidecar.provider_name
SEARCH_SIDECAR_PROVIDER_ORDER = (PORTABLE_LEXICAL_SIDECAR_PROVIDER,)

SEARCH_SIDECARS.register(
    PORTABLE_LEXICAL_SIDECAR_PROVIDER,
    _build_portable_lexical_sidecar,
)
