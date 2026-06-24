"""Retrieval-hit export contract: field names and the export-shaping helpers."""

from __future__ import annotations

from typing import Final
from collections.abc import Sequence
from typing import TYPE_CHECKING


RETRIEVAL_HIT_ID_FIELD: Final[str] = "id"

RETRIEVAL_HIT_CONTENT_FIELD: Final[str] = "content"

RETRIEVAL_HIT_SCORE_FIELD: Final[str] = "score"

RETRIEVAL_HIT_DOCUMENT_ID_FIELD: Final[str] = "document_id"

RETRIEVAL_HIT_DOCUMENT_KEY_FIELD: Final[str] = "document_key"

RETRIEVAL_HIT_COLLECTION_FIELD: Final[str] = "collection"

RETRIEVAL_HIT_NAMESPACE_FIELD: Final[str] = "namespace"

RETRIEVAL_HIT_TITLE_FIELD: Final[str] = "title"

RETRIEVAL_HIT_CHUNK_INDEX_FIELD: Final[str] = "chunk_index"

RETRIEVAL_HIT_SECTION_PATH_FIELD: Final[str] = "section_path"

RETRIEVAL_HIT_METADATA_FIELD: Final[str] = "metadata"

RETRIEVAL_HIT_CORE_FIELDS: Final[tuple[str, ...]] = (
    RETRIEVAL_HIT_ID_FIELD,
    RETRIEVAL_HIT_CONTENT_FIELD,
    RETRIEVAL_HIT_SCORE_FIELD,
)

RETRIEVAL_HIT_OPTIONAL_FIELDS: Final[tuple[str, ...]] = (
    RETRIEVAL_HIT_DOCUMENT_ID_FIELD,
    RETRIEVAL_HIT_DOCUMENT_KEY_FIELD,
    RETRIEVAL_HIT_COLLECTION_FIELD,
    RETRIEVAL_HIT_NAMESPACE_FIELD,
    RETRIEVAL_HIT_TITLE_FIELD,
    RETRIEVAL_HIT_CHUNK_INDEX_FIELD,
    RETRIEVAL_HIT_SECTION_PATH_FIELD,
    RETRIEVAL_HIT_METADATA_FIELD,
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
        if result.collection is not None:
            document[RETRIEVAL_HIT_COLLECTION_FIELD] = result.collection
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
