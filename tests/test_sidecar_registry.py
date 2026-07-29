from __future__ import annotations

import asyncio
from typing import Sequence

import pytest

from rag_core import Engine
from rag_core.config import IngestConfig
from rag_core.search.lexical_sidecar import (
    PortableLexicalSidecar,
    create_search_sidecar,
)
from rag_core.search.providers.registry import SEARCH_SIDECARS, ProviderRegistry
from rag_core.search.provider_protocols import SearchSidecar
from rag_core.search.request_models import SearchSidecarQuery
from rag_core.search.vector_models import SearchResult

from tests.support import (
    FakeEmbeddingProvider,
    FakeSparseEmbedder,
    RecordingVectorStore,
    make_test_config,
)


def test_builtin_portable_lexical_is_registered_and_constructible() -> None:
    assert isinstance(SEARCH_SIDECARS, ProviderRegistry)
    assert "portable_lexical" in SEARCH_SIDECARS
    assert isinstance(create_search_sidecar("portable_lexical"), PortableLexicalSidecar)


def test_create_search_sidecar_none_returns_none() -> None:
    """``None`` matches today's default of "no sidecar wired in"."""
    assert create_search_sidecar(None) is None


def test_create_search_sidecar_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="Unknown search_sidecar provider"):
        create_search_sidecar("not-a-real-sidecar")


def test_user_registered_sidecar_works_end_to_end() -> None:
    class _StubSidecar:
        def upsert_records(self, records: Sequence[object]) -> None:
            return None

        def delete_document(
            self,
            *,
            namespace: str,
            document_id: str,
            collection: str | None = None,
        ) -> None:
            return None

        async def search(self, query: SearchSidecarQuery) -> list[SearchResult]:
            return []

    name = "stub-test-sidecar"
    SEARCH_SIDECARS.register(name, lambda **kwargs: _StubSidecar())
    try:
        sidecar: SearchSidecar | None = create_search_sidecar(name)
        assert isinstance(sidecar, _StubSidecar)
    finally:
        SEARCH_SIDECARS.unregister(name)


def test_rag_core_resolves_lexical_search_provider_from_config() -> None:
    """``Config.ingest.lexical_search_provider`` wires lexical search."""
    base = make_test_config(
        qdrant_collection="rag_core_lexical_search_provider_test",
        embedding_dimensions=4,
    )
    config = type(base)(
        qdrant=base.qdrant,
        embedding=base.embedding,
        reranker=base.reranker,
        chunking=base.chunking,
        ingest=IngestConfig(
            processing_version=base.ingest.processing_version,
            source_type=base.ingest.source_type,
            enable_lexical_search=base.ingest.enable_lexical_search,
            manifest_directory=base.ingest.manifest_directory,
            lexical_search_provider="portable_lexical",
        ),
        policy=base.policy,
    )
    core = Engine(
        config,
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=RecordingVectorStore(),
    )
    try:
        assert isinstance(core._sidecar, PortableLexicalSidecar)
    finally:
        asyncio.run(core.close())
