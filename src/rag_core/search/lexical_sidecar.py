"""Portable lexical sidecar for exact or trigram-style matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from rag_core.search.lexical_sidecar_matching import (
    LexicalMatch,
    match_lexical_result,
    normalized_lexical_query,
)
from rag_core.search.providers.registry import SEARCH_SIDECARS
from rag_core.search.types import SearchResult, SearchSidecar, SearchSidecarQuery


@dataclass(frozen=True)
class LexicalSidecarRecord:
    """Portable record shape for sidecar lookups."""

    namespace: str
    result: SearchResult


class PortableLexicalSidecar(SearchSidecar):
    """Small in-memory sidecar for exact and trigram-style retrieval."""

    def __init__(
        self,
        records: list[LexicalSidecarRecord],
        *,
        trigram_threshold: float = 0.35,
    ) -> None:
        self._records = list(records)
        self._trigram_threshold = trigram_threshold

    def upsert_records(self, records: Sequence[object]) -> None:
        existing = {
            (record.namespace, record.result.id): record
            for record in self._records
        }
        for record in records:
            if not isinstance(record, LexicalSidecarRecord):
                continue
            existing[(record.namespace, record.result.id)] = record
        self._records = list(existing.values())

    def delete_document(
        self,
        *,
        namespace: str,
        document_id: str,
        corpus_id: str | None = None,
    ) -> None:
        self._records = [
            record
            for record in self._records
            if not (
                record.namespace == namespace
                and record.result.document_id == document_id
                and (corpus_id is None or record.result.corpus_id == corpus_id)
            )
        ]

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


SEARCH_SIDECARS.register("portable_lexical", _build_portable_lexical_sidecar)
