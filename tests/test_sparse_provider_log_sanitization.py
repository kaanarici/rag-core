from __future__ import annotations


import pytest

from rag_core.search.providers import sparse
from tests.support import TEST_API_SECRET

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

    def query_embed(self, texts: list[str]) -> list[_RawSparseVector]:
        return self.embed(texts)


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








def test_sparse_model_env_defaults_are_resolved_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPARSE_EMBEDDING_MODEL_BM25", "env-bm25")
    monkeypatch.setattr(
        sparse,
        "_import_sparse_text_embedding",
        lambda: _SuccessfulSparseTextEmbedding,
    )

    embedder = sparse.FastEmbedSparseEmbedder()

    assert embedder._bm25_model_name == "env-bm25"


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
