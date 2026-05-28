from __future__ import annotations

from pathlib import Path

from rag_core.events.export import to_retrieval_hits
from rag_core.events.retrieval_hit_fields import (
    RETRIEVAL_HIT_CONTENT_FIELD,
    RETRIEVAL_HIT_CORPUS_ID_FIELD,
    RETRIEVAL_HIT_DOCUMENT_ID_FIELD,
    RETRIEVAL_HIT_DOCUMENT_KEY_FIELD,
    RETRIEVAL_HIT_ID_FIELD,
    RETRIEVAL_HIT_METADATA_FIELD,
    RETRIEVAL_HIT_NAMESPACE_FIELD,
    RETRIEVAL_HIT_SCORE_FIELD,
    RETRIEVAL_HIT_SECTION_PATH_FIELD,
    RETRIEVAL_HIT_CHUNK_INDEX_FIELD,
    RETRIEVAL_HIT_TITLE_FIELD,
)
from rag_core.search import SearchResult
from tests.support import make_search_result


def test_to_retrieval_hits_maps_search_result_fields() -> None:
    hit = make_search_result(
        document_id="doc-1",
        document_key="docs/guide.md",
        score=0.91,
        namespace="acme",
        corpus_id="help",
        title="Guide",
        chunk_index=2,
        section_path="Guide > Billing",
        metadata={"team": "support"},
    )

    exported = to_retrieval_hits([hit])

    assert exported == [
        {
            RETRIEVAL_HIT_ID_FIELD: hit.id,
            RETRIEVAL_HIT_CONTENT_FIELD: hit.text,
            RETRIEVAL_HIT_SCORE_FIELD: 0.91,
            RETRIEVAL_HIT_DOCUMENT_ID_FIELD: "doc-1",
            RETRIEVAL_HIT_DOCUMENT_KEY_FIELD: "docs/guide.md",
            RETRIEVAL_HIT_CORPUS_ID_FIELD: "help",
            RETRIEVAL_HIT_NAMESPACE_FIELD: "acme",
            RETRIEVAL_HIT_TITLE_FIELD: "Guide",
            RETRIEVAL_HIT_CHUNK_INDEX_FIELD: 2,
            RETRIEVAL_HIT_SECTION_PATH_FIELD: "Guide > Billing",
            RETRIEVAL_HIT_METADATA_FIELD: {"team": "support"},
        }
    ]


def test_to_retrieval_hits_omits_empty_optional_fields() -> None:
    hit = SearchResult(
        id="result-1",
        text="fox query context",
        score=0.9,
        content_type="document",
        source_type="file",
        document_id="doc-1",
    )

    exported = to_retrieval_hits([hit])

    assert exported[0][RETRIEVAL_HIT_ID_FIELD] == hit.id
    assert exported[0][RETRIEVAL_HIT_CONTENT_FIELD] == hit.text
    assert RETRIEVAL_HIT_METADATA_FIELD not in exported[0]
    assert RETRIEVAL_HIT_NAMESPACE_FIELD not in exported[0]
    assert RETRIEVAL_HIT_CORPUS_ID_FIELD not in exported[0]
    assert RETRIEVAL_HIT_TITLE_FIELD not in exported[0]


def test_events_export_uses_curated_search_result_import() -> None:
    source = Path("src/rag_core/events/export.py").read_text(encoding="utf-8")

    assert "from rag_core.search import SearchResult" in source
    assert "rag_core.search.vector_models" not in source
