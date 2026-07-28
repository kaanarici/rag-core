from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from rag_core.documents.contextualizer import ChunkContextRequest
from rag_core.documents.contextualizer_adapters import AnthropicChunkContextualizer
from tests.support import assert_caplog_omits_private


class ProviderSecretError(RuntimeError):
    pass


class _FailingMessages:
    async def create(self, **_: Any) -> object:
        raise ProviderSecretError("raw provider detail: api key sk-test-secret")


class _FailingClient:
    def __init__(self) -> None:
        self.messages = _FailingMessages()


def test_anthropic_contextualizer_failure_log_is_sanitized(
    caplog: pytest.LogCaptureFixture,
) -> None:
    contextualizer = AnthropicChunkContextualizer(
        model="claude-test",
        client=_FailingClient(),
    )
    request = ChunkContextRequest(
        document_markdown="private document body",
        document_filename="sensitive-roadmap.md",
        chunk_text="private chunk text",
        chunk_index=3,
        total_chunks=9,
    )

    with caplog.at_level(
        logging.DEBUG, logger="rag_core.documents.contextualizer_adapters"
    ):
        with pytest.raises(ProviderSecretError):
            asyncio.run(contextualizer.contextualize(request))

    assert "claude-test" in caplog.text
    assert "chunk 3" in caplog.text
    assert "ProviderSecretError" in caplog.text
    assert_caplog_omits_private(
        caplog,
        "raw provider detail",
        "sensitive-roadmap.md",
        "private document body",
        "private chunk text",
    )
