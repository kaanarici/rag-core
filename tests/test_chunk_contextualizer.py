from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError

import pytest

from rag_core.documents.contextualizer import (
    ChunkContextRequest,
    ChunkContextualizer,
    NoOpContextualizer,
)


def _make_request(
    *,
    chunk_index: int = 0,
    chunk_text: str = "alpha chunk",
    total_chunks: int = 2,
    document_markdown: str = "doc body",
    document_filename: str = "doc.md",
) -> ChunkContextRequest:
    return ChunkContextRequest(
        document_markdown=document_markdown,
        document_filename=document_filename,
        chunk_text=chunk_text,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
    )


def test_chunk_context_request_carries_all_fields() -> None:
    request = _make_request(chunk_index=3, total_chunks=7)
    assert request.chunk_index == 3
    assert request.total_chunks == 7
    assert request.document_filename == "doc.md"
    assert request.chunk_text == "alpha chunk"


def test_chunk_context_request_is_frozen() -> None:
    request = _make_request()
    with pytest.raises(FrozenInstanceError):
        request.chunk_index = 99  # type: ignore[misc]


def test_noop_contextualizer_returns_empty_string_and_satisfies_protocol() -> None:
    contextualizer = NoOpContextualizer()
    assert isinstance(contextualizer, ChunkContextualizer)
    assert contextualizer.contextualizer_id == "noop"
    assert asyncio.run(contextualizer.contextualize(_make_request())) == ""
