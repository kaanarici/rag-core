from __future__ import annotations

from rag_core.events.export import to_retrieval_hits
from rag_core.search.vector_models import SearchResult
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
            "id": hit.id,
            "content": hit.text,
            "score": 0.91,
            "document_id": "doc-1",
            "document_key": "docs/guide.md",
            "corpus_id": "help",
            "namespace": "acme",
            "title": "Guide",
            "chunk_index": 2,
            "section_path": "Guide > Billing",
            "metadata": {"team": "support"},
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

    assert exported[0]["id"] == hit.id
    assert exported[0]["content"] == hit.text
    assert "metadata" not in exported[0]
    assert "namespace" not in exported[0]
    assert "corpus_id" not in exported[0]
    assert "title" not in exported[0]
