"""Portable retrieval-hit export for observability adapters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from rag_core.events.retrieval_hit_fields import (
    RETRIEVAL_HIT_CHUNK_INDEX_FIELD,
    RETRIEVAL_HIT_CONTENT_FIELD,
    RETRIEVAL_HIT_CORPUS_ID_FIELD,
    RETRIEVAL_HIT_DOCUMENT_ID_FIELD,
    RETRIEVAL_HIT_DOCUMENT_KEY_FIELD,
    RETRIEVAL_HIT_ID_FIELD,
    RETRIEVAL_HIT_METADATA_FIELD,
    RETRIEVAL_HIT_NAMESPACE_FIELD,
    RETRIEVAL_HIT_SCORE_FIELD,
    RETRIEVAL_HIT_SECTION_PATH_FIELD,
    RETRIEVAL_HIT_TITLE_FIELD,
)

if TYPE_CHECKING:
    from rag_core.search import SearchResult


def to_retrieval_hits(results: Sequence[SearchResult]) -> list[dict[str, object]]:
    """Map ``SearchResult`` rows to a stable JSON shape for tracing SDKs.

    The payload aligns with common ``gen_ai.retrieval.documents`` conventions and
    ``scored_chunks``-style fields (``id``, ``content``, ``score``, document
    locators).
    """

    hits: list[dict[str, object]] = []
    for result in results:
        document: dict[str, object] = {
            RETRIEVAL_HIT_ID_FIELD: result.id,
            RETRIEVAL_HIT_CONTENT_FIELD: result.text,
            RETRIEVAL_HIT_SCORE_FIELD: result.score,
        }
        if result.document_id is not None:
            document[RETRIEVAL_HIT_DOCUMENT_ID_FIELD] = result.document_id
        if result.document_key is not None:
            document[RETRIEVAL_HIT_DOCUMENT_KEY_FIELD] = result.document_key
        if result.corpus_id is not None:
            document[RETRIEVAL_HIT_CORPUS_ID_FIELD] = result.corpus_id
        if result.namespace is not None:
            document[RETRIEVAL_HIT_NAMESPACE_FIELD] = result.namespace
        if result.title is not None:
            document[RETRIEVAL_HIT_TITLE_FIELD] = result.title
        if result.chunk_index is not None:
            document[RETRIEVAL_HIT_CHUNK_INDEX_FIELD] = result.chunk_index
        if result.section_path is not None:
            document[RETRIEVAL_HIT_SECTION_PATH_FIELD] = result.section_path
        if result.metadata:
            document[RETRIEVAL_HIT_METADATA_FIELD] = {
                key: value
                for key, value in result.metadata.items()
                if value is not None
            }
            if not document[RETRIEVAL_HIT_METADATA_FIELD]:
                document.pop(RETRIEVAL_HIT_METADATA_FIELD)
        hits.append(document)
    return hits
