from __future__ import annotations

import rag_core
import rag_core.search as search
from rag_core.search.context_pack import (
    ContextSnippet,
    SourceLocator,
    SourcePreview,
    SourceReference,
)


def test_context_pack_citation_primitives_are_public_imports() -> None:
    assert rag_core.ContextSnippet is ContextSnippet
    assert rag_core.SourceLocator is SourceLocator
    assert rag_core.SourcePreview is SourcePreview
    assert rag_core.SourceReference is SourceReference

    assert search.ContextSnippet is ContextSnippet
    assert search.SourceLocator is SourceLocator
    assert search.SourcePreview is SourcePreview
    assert search.SourceReference is SourceReference


def test_source_preview_remains_a_small_app_facing_payload() -> None:
    preview = SourcePreview(
        citation_id="billing#chunk-0",
        title="billing.md",
        locator_label="page 2, chunk 0",
        document_id="billing",
        corpus_id="help",
        source_hash="sha256:abc",
    )

    assert preview.as_text() == "[billing#chunk-0] billing.md (page 2, chunk 0)"
    assert preview.to_payload()["source_hash"] == "sha256:abc"
