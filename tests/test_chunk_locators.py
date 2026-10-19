"""Locator reliability assertions for the chunker family.

These tests pin the span-persistence contract: a chunk's char span either
reproduces the chunk text verbatim from the source, or the chunk metadata
carries ``offset_reconstruction='unreliable'`` (without raising) so
EvidenceSpan resolvers refuse it.
"""

from __future__ import annotations

from time import perf_counter

from rag_core.config import MARKDOWN_CHUNKING_STRATEGY
from rag_core.config import ChunkingConfig
from rag_core._engine.core_prepare import prepare_text_chunks
from rag_core.core_models import estimate_token_count
from rag_core.documents.chunking.budget import token_budget_for_char_limit
from rag_core.documents.chunking.markdown import MarkdownChunker
from rag_core.documents.chunking.protocol import ChunkConfig


def _config() -> ChunkConfig:
    return ChunkConfig(
        max_chars=2000,
        overlap=200,
        strategy=MARKDOWN_CHUNKING_STRATEGY,
    )


def _reliable_span(chunk: object) -> tuple[int, int]:
    start = getattr(chunk, "start_char")
    end = getattr(chunk, "end_char")
    assert isinstance(start, int)
    assert isinstance(end, int)
    return start, end


def test_markdown_chunker_records_reliable_offsets_for_recoverable_text() -> None:
    text = "# Title\n\nFirst paragraph.\n\n## Heading\n\nSecond paragraph.\n"
    chunks = MarkdownChunker().chunk(text, _config())

    assert chunks
    for chunk in chunks:
        start, end = _reliable_span(chunk)
        assert start >= 0
        assert end > start
        assert "offset_reconstruction" not in chunk.metadata


def test_markdown_chunker_spans_reproduce_text_with_default_overlap() -> None:
    # Overlapped chunks must remain exact source slices: every span that is
    # not flagged unreliable reproduces the chunk text verbatim.
    paragraphs = [
        f"Paragraph {i}: " + ("alpha beta gamma delta epsilon zeta " * 12).strip()
        for i in range(20)
    ]
    text = "# Title\n\n" + "\n\n".join(paragraphs)
    chunks = MarkdownChunker().chunk(text, _config())

    assert len(chunks) > 1
    for chunk in chunks:
        assert "offset_reconstruction" not in chunk.metadata
        assert text[chunk.start_char : chunk.end_char] == chunk.text


def test_markdown_chunker_spans_survive_trailing_whitespace() -> None:
    # Trailing whitespace inside paragraphs must not silently corrupt spans:
    # each chunk either reproduces its text exactly or is flagged unreliable.
    paragraphs = [
        f"Paragraph {i}: " + ("alpha beta gamma delta epsilon zeta " * 12)
        for i in range(20)
    ]
    text = "# Title\n\n" + "\n\n".join(paragraphs)
    chunks = MarkdownChunker().chunk(text, _config())

    assert chunks
    for chunk in chunks:
        if chunk.metadata.get("offset_reconstruction") == "unreliable":
            continue
        assert text[chunk.start_char : chunk.end_char] == chunk.text


def test_markdown_chunker_hard_splits_single_oversized_part() -> None:
    chunks = prepare_text_chunks(
        "# T\n\n" + ("x" * 5005),
        filename="long.md",
        mime_type="text/markdown",
        chunking_config=ChunkingConfig(max_chars=1000, overlap=0),
    )

    assert chunks
    assert max(len(chunk.text) for chunk in chunks) <= 1000


def test_code_chunker_hard_splits_single_oversized_line() -> None:
    chunks = prepare_text_chunks(
        'value = "' + ("a" * 5005) + '"',
        filename="long.py",
        mime_type="text/x-python",
        chunking_config=ChunkingConfig(max_chars=1000, overlap=0),
    )

    assert chunks
    assert max(len(chunk.text) for chunk in chunks) <= 1000


def test_builtin_chunkers_enforce_dense_token_budget() -> None:
    dense = ("界" * 350) + ("+-=" * 50)
    max_chars = 400
    max_tokens = token_budget_for_char_limit(max_chars)

    for strategy, filename, mime_type in (
        ("markdown", "dense.md", "text/markdown"),
        ("auto", "dense.txt", "text/plain"),
        ("auto", "dense.py", "text/x-python"),
    ):
        chunks = prepare_text_chunks(
            dense,
            filename=filename,
            mime_type=mime_type,
            chunking_config=ChunkingConfig(
                strategy=strategy,  # type: ignore[arg-type]
                max_chars=max_chars,
                overlap=200,
            ),
        )

        assert len(chunks) > 1
        assert all(estimate_token_count(chunk.text) <= max_tokens for chunk in chunks)


def test_markdown_overlap_transition_rechecks_combined_budget() -> None:
    text = ("界" * 350) + "\n\n" + ("語" * 20)
    chunks = MarkdownChunker().chunk(
        text,
        ChunkConfig(max_chars=400, overlap=200, strategy="markdown"),
    )

    assert max(estimate_token_count(chunk.text) for chunk in chunks) <= 100
    assert all(
        text[chunk.start_char : chunk.end_char] == chunk.text for chunk in chunks
    )


def test_dense_unicode_splitting_is_near_linear_at_one_million_characters() -> None:
    config = ChunkingConfig(max_chars=400, overlap=200)

    def elapsed_for(size: int) -> float:
        started = perf_counter()
        chunks = prepare_text_chunks(
            "界" * size,
            filename="dense.md",
            mime_type="text/markdown",
            chunking_config=config,
        )
        elapsed = perf_counter() - started
        assert chunks
        assert max(estimate_token_count(chunk.text) for chunk in chunks) <= 100
        return elapsed

    small_elapsed = elapsed_for(100_000)
    large_elapsed = elapsed_for(1_000_000)

    assert large_elapsed < 5.0
    assert large_elapsed <= (small_elapsed * 15) + 0.25
