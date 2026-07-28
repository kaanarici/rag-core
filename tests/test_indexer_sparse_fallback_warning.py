from __future__ import annotations

import logging

import pytest

from rag_core.search.indexer_prepare import embed_sparse_channels
from rag_core.search.sparse_channels import PRIMARY_SPARSE_CHANNEL
from rag_core.search.vector_models import SparseVector
from tests.support import FakeSparseEmbedder, assert_caplog_omits_private


class ProviderSecretError(RuntimeError):
    pass


class _FailingMultiSparseEmbedder(FakeSparseEmbedder):
    provider_name = "fake-sparse"

    def embed_texts_multi(self, texts: list[str]) -> list[dict[str, SparseVector]]:
        self.embed_texts_multi_calls.append(list(texts))
        raise ProviderSecretError("raw provider detail: api key sk-test-secret")


def test_sparse_multi_failure_warning_is_sanitized(
    caplog: pytest.LogCaptureFixture,
) -> None:
    embedder = _FailingMultiSparseEmbedder()

    with caplog.at_level(
        logging.WARNING, logger="rag_core.search.indexer_prepare"
    ):
        sparse_channels = embed_sparse_channels(
            sparse_embedder=embedder,
            texts=["fox query"],
            expected_count=1,
        )

    assert embedder.embed_texts_multi_calls == [["fox query"]]
    assert embedder.embed_texts_calls == [["fox query"]]
    assert [set(channels) for channels in sparse_channels] == [
        {PRIMARY_SPARSE_CHANNEL}
    ]
    assert sparse_channels[0][PRIMARY_SPARSE_CHANNEL].indices == [3, 4]
    assert "ProviderSecretError" in caplog.text
    assert_caplog_omits_private(caplog, "raw provider detail")
