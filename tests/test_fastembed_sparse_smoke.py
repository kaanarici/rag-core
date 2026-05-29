"""FastEmbed sparse inference smoke — real local model, no API keys."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.integration]


def test_fastembed_sparse_embedder_initialization_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_core.search.providers import sparse as sparse_module
    from rag_core.search.providers.sparse import FastEmbedSparseEmbedder

    def fail_import() -> object:
        raise AssertionError("FastEmbed should not load until sparse embedding is used")

    monkeypatch.setattr(sparse_module, "_import_sparse_text_embedding", fail_import)

    embedder = FastEmbedSparseEmbedder(enable_splade=False)

    assert embedder.diagnostics()["bm25_load_status"] == "not_loaded"
    assert embedder.model_name == "Qdrant/bm25"


def test_fastembed_sparse_embedder_returns_non_empty_indices() -> None:
    if os.environ.get("RAG_CORE_SKIP_FASTEMBED_DOWNLOAD") == "1":
        pytest.skip("RAG_CORE_SKIP_FASTEMBED_DOWNLOAD=1 skips FastEmbed model download")

    from rag_core.search.providers.sparse import FastEmbedSparseEmbedder

    embedder = FastEmbedSparseEmbedder(enable_splade=False)
    vectors = embedder.embed_texts(["invoice billing webhook", "pagination cursor limit"])

    assert len(vectors) == 2
    assert all(vector.indices for vector in vectors)
    assert all(vector.values for vector in vectors)
    assert len(vectors[0].indices) == len(vectors[0].values)

    query = embedder.embed_query("How do webhooks sign payloads?")
    assert query.indices
    assert query.values
