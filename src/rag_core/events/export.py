"""Portable retrieval-hit export for observability adapters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_core.search.vector_models import SearchResult


def to_retrieval_hits(results: Sequence[SearchResult]) -> list[dict[str, object]]:
    """Map ``SearchResult`` rows to a stable JSON shape for tracing SDKs.

    The payload aligns with common ``gen_ai.retrieval.documents`` conventions and
    Ragie-style ``scored_chunks`` (``id``, ``text``, ``score``, document locators).
    """

    hits: list[dict[str, object]] = []
    for result in results:
        document: dict[str, object] = {
            "id": result.id,
            "content": result.text,
            "score": result.score,
        }
        if result.document_id is not None:
            document["document_id"] = result.document_id
        if result.document_key is not None:
            document["document_key"] = result.document_key
        if result.corpus_id is not None:
            document["corpus_id"] = result.corpus_id
        if result.namespace is not None:
            document["namespace"] = result.namespace
        if result.title is not None:
            document["title"] = result.title
        if result.chunk_index is not None:
            document["chunk_index"] = result.chunk_index
        if result.section_path is not None:
            document["section_path"] = result.section_path
        if result.metadata:
            document["metadata"] = {
                key: value for key, value in result.metadata.items() if value is not None
            }
            if not document["metadata"]:
                document.pop("metadata")
        hits.append(document)
    return hits
