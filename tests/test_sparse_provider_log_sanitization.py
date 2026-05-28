from __future__ import annotations

import logging

import pytest

from rag_core.search.providers import sparse
from rag_core.search.providers.sparse import SPARSE_LOAD_FAILED, SPARSE_LOAD_LOADED
from rag_core.search.sparse_channels import (
    PRIMARY_SPARSE_CHANNEL,
    SECONDARY_SPARSE_CHANNEL,
)
from rag_core.search.types import SparseVector
from tests.support import TEST_API_SECRET, assert_caplog_omits_private

LOGGER_NAME = "rag_core.search.providers.sparse"
PRIVATE_BM25_MODEL = "/Users/person/private-bm25-sk-test-secret"
PRIVATE_SPLADE_MODEL = "/Users/person/private-splade-sk-test-secret"


class ProviderSecretError(RuntimeError):
    pass


class _RawSparseVector:
    def __init__(self, index: int) -> None:
        self.indices = [index]
        self.values = [1.0]


class _FakeSparseTextEmbedding:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        if model_name == PRIVATE_SPLADE_MODEL:
            raise ProviderSecretError(
                "raw fastembed setup detail for /Users/person with sk-test-secret"
            )

    def embed(self, texts: list[str]) -> list[_RawSparseVector]:
        return [_RawSparseVector(idx + 1) for idx, _text in enumerate(texts)]


class _SuccessfulSparseTextEmbedding(_FakeSparseTextEmbedding):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name


class _EmptySparseTextEmbedding(_FakeSparseTextEmbedding):
    def embed(self, texts: list[str]) -> list[_RawSparseVector]:
        return []


class _Bm25OkSpladeEmptySparseTextEmbedding(_FakeSparseTextEmbedding):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def embed(self, texts: list[str]) -> list[_RawSparseVector]:
        if self.model_name == PRIVATE_SPLADE_MODEL:
            return []
        return [_RawSparseVector(idx + 1) for idx, _text in enumerate(texts)]


def _joined_messages(caplog: pytest.LogCaptureFixture) -> str:
    return "\n".join(record.getMessage() for record in caplog.records)


def test_splade_count_mismatch_logs_sanitized_warning_and_uses_bm25_fallback(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sparse,
        "_import_sparse_text_embedding",
        lambda: _Bm25OkSpladeEmptySparseTextEmbedding,
    )
    embedder = sparse.FastEmbedSparseEmbedder(
        bm25_model_name=PRIVATE_BM25_MODEL,
        splade_model_name=PRIVATE_SPLADE_MODEL,
    )

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        channels = embedder.embed_texts_multi(["private one", "private two"])

    assert channels == [
        {PRIMARY_SPARSE_CHANNEL: SparseVector(indices=[1], values=[1.0])},
        {PRIMARY_SPARSE_CHANNEL: SparseVector(indices=[2], values=[1.0])},
    ]
    message = _joined_messages(caplog)
    assert "provider=fastembed" in message
    assert "backend=fastembed" not in message
    assert f"channel={SECONDARY_SPARSE_CHANNEL}" in message
    assert f"fallback_channel={PRIMARY_SPARSE_CHANNEL}" in message
    assert "expected=2 actual=0" in message
    assert_caplog_omits_private(
        caplog,
        PRIVATE_BM25_MODEL,
        PRIVATE_SPLADE_MODEL,
        "/Users/person",
        "raw fastembed setup detail",
    )


def test_splade_load_failure_log_is_sanitized_and_uses_bm25_fallback(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sparse,
        "_import_sparse_text_embedding",
        lambda: _FakeSparseTextEmbedding,
    )
    embedder = sparse.FastEmbedSparseEmbedder(
        bm25_model_name=PRIVATE_BM25_MODEL,
        splade_model_name=PRIVATE_SPLADE_MODEL,
    )

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        channels = embedder.embed_texts_multi(["private text"])

    assert channels == [
        {PRIMARY_SPARSE_CHANNEL: SparseVector(indices=[1], values=[1.0])}
    ]
    assert embedder.diagnostics()["splade_enabled"] is False
    assert embedder.diagnostics()["splade_load_status"] == SPARSE_LOAD_FAILED
    message = _joined_messages(caplog)
    assert "provider=fastembed" in message
    assert "backend=fastembed" not in message
    assert f"channel={SECONDARY_SPARSE_CHANNEL}" in message
    assert f"fallback_channel={PRIMARY_SPARSE_CHANNEL}" in message
    assert "error_type=ProviderSecretError" in message
    assert "Loaded sparse model" not in message
    assert_caplog_omits_private(
        caplog,
        PRIVATE_BM25_MODEL,
        PRIVATE_SPLADE_MODEL,
        "/Users/person",
        "raw fastembed setup detail",
    )


def test_splade_load_success_log_omits_configured_model_identifier(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sparse,
        "_import_sparse_text_embedding",
        lambda: _SuccessfulSparseTextEmbedding,
    )
    embedder = sparse.FastEmbedSparseEmbedder(
        bm25_model_name=PRIVATE_BM25_MODEL,
        splade_model_name=PRIVATE_SPLADE_MODEL,
    )

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        channels = embedder.embed_texts_multi(["private text"])

    assert set(channels[0]) == {PRIMARY_SPARSE_CHANNEL, SECONDARY_SPARSE_CHANNEL}
    assert embedder.diagnostics()["splade_enabled"] is True
    assert embedder.diagnostics()["splade_load_status"] == SPARSE_LOAD_LOADED
    message = _joined_messages(caplog)
    assert "provider=fastembed" in message
    assert "backend=fastembed" not in message
    assert f"channel={SECONDARY_SPARSE_CHANNEL}" in message
    assert "fallback_channel=" not in message
    assert "error_type=" not in message
    assert_caplog_omits_private(
        caplog,
        PRIVATE_BM25_MODEL,
        PRIVATE_SPLADE_MODEL,
        "/Users/person",
        "raw fastembed setup detail",
    )


def test_sparse_model_env_defaults_are_resolved_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPARSE_EMBEDDING_MODEL_BM25", "env-bm25")
    monkeypatch.setenv("SPARSE_EMBEDDING_MODEL_SPLADE", "env-splade")
    monkeypatch.setattr(
        sparse,
        "_import_sparse_text_embedding",
        lambda: _SuccessfulSparseTextEmbedding,
    )

    embedder = sparse.FastEmbedSparseEmbedder()

    assert embedder._bm25_model_name == "env-bm25"
    assert embedder._splade_model_name == "env-splade"


def test_sparse_bm25_default_keeps_base_env_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SPARSE_EMBEDDING_MODEL_BM25", raising=False)
    monkeypatch.setenv("SPARSE_EMBEDDING_MODEL", "base-env-bm25")
    monkeypatch.setattr(
        sparse,
        "_import_sparse_text_embedding",
        lambda: _SuccessfulSparseTextEmbedding,
    )

    embedder = sparse.FastEmbedSparseEmbedder()

    assert embedder._bm25_model_name == "base-env-bm25"


def test_single_query_sparse_count_mismatch_raises_sanitized_contract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sparse,
        "_import_sparse_text_embedding",
        lambda: _EmptySparseTextEmbedding,
    )
    embedder = sparse.FastEmbedSparseEmbedder(
        bm25_model_name=PRIVATE_BM25_MODEL,
        splade_model_name=PRIVATE_SPLADE_MODEL,
    )

    with pytest.raises(ValueError) as exc_info:
        embedder.embed_query("private text " + PRIVATE_BM25_MODEL)

    message = str(exc_info.value)
    assert message == (
        "FastEmbedSparseEmbedder provider contract violation: "
        "expected 1 sparse vectors, got 0"
    )
    assert PRIVATE_BM25_MODEL not in message
    assert PRIVATE_SPLADE_MODEL not in message
    assert "/Users/person" not in message
    assert TEST_API_SECRET not in message
