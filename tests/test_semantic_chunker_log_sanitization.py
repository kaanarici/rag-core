from __future__ import annotations

import asyncio
import logging

import pytest

from rag_core.core_models import PreparedChunk
from rag_core.documents.chunking.protocol import ChunkConfig
from rag_core.documents.chunking.semantic import (
    SemanticChunker,
    _LocalSemanticEmbedder,
)
from tests.support import assert_caplog_omits_private

_TEXT = (
    "Alpha topic starts with setup. "
    "Beta topic continues with implementation detail. "
    "Gamma topic closes with validation."
)
_CONFIG = ChunkConfig(max_chars=52, overlap=0)


class LocalModelSetupError(RuntimeError):
    pass


class ProviderEmbeddingError(RuntimeError):
    pass


class _FailingLocalSemanticEmbedder:
    async def embed_many(self, sentences: list[str]) -> list[list[float]]:
        assert sentences
        raise ProviderEmbeddingError(
            "raw local model embed detail with api key sk-test-secret"
        )


def _heuristic_texts() -> list[str]:
    return [
        chunk.text
        for chunk in SemanticChunker(enable_local_model=False).chunk(_TEXT, _CONFIG)
    ]


def _assert_heuristic_fallback(chunks: list[PreparedChunk]) -> None:
    assert [chunk.text for chunk in chunks] == _heuristic_texts()
    assert {chunk.chunking_strategy for chunk in chunks} == {"semantic_heuristic"}


def test_local_semantic_embedder_setup_warning_is_sanitized(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_setup(
        cls: type[_LocalSemanticEmbedder], model_name: str
    ) -> _LocalSemanticEmbedder:
        assert cls is _LocalSemanticEmbedder
        assert model_name == "safe-local-model"
        raise LocalModelSetupError(
            "raw local setup detail with api key sk-test-secret"
        )

    monkeypatch.setattr(_LocalSemanticEmbedder, "get", classmethod(fail_setup))
    chunker = SemanticChunker(
        enable_local_model=True,
        local_model_name="safe-local-model",
    )

    with caplog.at_level(
        logging.WARNING, logger="rag_core.documents.chunking.semantic"
    ):
        chunks = asyncio.run(chunker.chunk_async(_TEXT, _CONFIG))

    _assert_heuristic_fallback(chunks)
    assert "safe-local-model" in caplog.text
    assert "LocalModelSetupError" in caplog.text
    assert_caplog_omits_private(caplog, "raw local setup detail")


def test_local_semantic_embedding_failure_warning_includes_model_context(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def get_embedder(
        cls: type[_LocalSemanticEmbedder], model_name: str
    ) -> _FailingLocalSemanticEmbedder:
        assert cls is _LocalSemanticEmbedder
        assert model_name == "safe-local-model"
        return _FailingLocalSemanticEmbedder()

    monkeypatch.setattr(_LocalSemanticEmbedder, "get", classmethod(get_embedder))
    chunker = SemanticChunker(
        enable_local_model=True,
        local_model_name="safe-local-model",
    )

    with caplog.at_level(
        logging.WARNING, logger="rag_core.documents.chunking.semantic"
    ):
        chunks = asyncio.run(chunker.chunk_async(_TEXT, _CONFIG))

    _assert_heuristic_fallback(chunks)
    assert "safe-local-model" in caplog.text
    assert "ProviderEmbeddingError" in caplog.text
    assert_caplog_omits_private(caplog, "raw local model embed detail")


def test_semantic_embedding_failure_warning_is_sanitized(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def fail_embed(sentences: list[str]) -> list[list[float]]:
        assert sentences
        raise ProviderEmbeddingError(
            "raw provider/model exception text with api key sk-test-secret"
        )

    chunker = SemanticChunker(embed_fn=fail_embed)

    with caplog.at_level(
        logging.WARNING, logger="rag_core.documents.chunking.semantic"
    ):
        chunks = asyncio.run(chunker.chunk_async(_TEXT, _CONFIG))

    _assert_heuristic_fallback(chunks)
    assert "ProviderEmbeddingError" in caplog.text
    assert_caplog_omits_private(caplog, "raw provider/model exception text")
