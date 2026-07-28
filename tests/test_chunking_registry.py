from __future__ import annotations

from typing import Any, List

import pytest

from rag_core.core_models import PreparedChunk
from rag_core.config import (
    CODE_CHUNKING_STRATEGY,
    MARKDOWN_CHUNKING_STRATEGY,
    SEMANTIC_CHUNKING_STRATEGY,
)
from rag_core.documents import CHUNKING_STRATEGIES, create_chunking_strategy
from rag_core.documents.chunking.code import CodeChunker
from rag_core.documents.chunking.markdown import MarkdownChunker
from rag_core.documents.chunking.protocol import ChunkConfig, ChunkingStrategy
from rag_core.documents.chunking.router import chunk_text
from rag_core.documents.chunking.semantic import SemanticChunker
from rag_core.documents.markdown_headings import parse_atx_heading
from rag_core.search.providers.registry import ProviderRegistry


def test_chunking_strategies_is_typed_provider_registry() -> None:
    assert isinstance(CHUNKING_STRATEGIES, ProviderRegistry)


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [
        (MARKDOWN_CHUNKING_STRATEGY, MarkdownChunker),
        (SEMANTIC_CHUNKING_STRATEGY, SemanticChunker),
        (CODE_CHUNKING_STRATEGY, CodeChunker),
    ],
)
def test_builtin_chunking_strategy_resolves_to_concrete_type(
    name: str, expected_type: type
) -> None:
    assert name in CHUNKING_STRATEGIES
    assert isinstance(create_chunking_strategy(name), expected_type)


def test_create_chunking_strategy_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="Unknown chunking_strategy provider"):
        create_chunking_strategy("not-a-real-strategy")


def test_atx_parser_only_consumes_whitespace_separated_closing_hashes() -> None:
    csharp = parse_atx_heading("## C#")
    closed = parse_atx_heading("## Release notes ###   ")
    compact_hashes = parse_atx_heading("## Heading###")

    assert csharp is not None and csharp.title == "C#"
    assert closed is not None and closed.title == "Release notes"
    assert compact_hashes is not None and compact_hashes.title == "Heading###"


def test_user_registered_chunking_strategy_works_end_to_end() -> None:
    class _CountingChunker:
        def chunk(self, text: str, config: ChunkConfig) -> List[PreparedChunk]:
            return [
                PreparedChunk(
                    chunk_index=0,
                    text=text,
                    embedding_text=text,
                    word_count=len(text.split()),
                    start_char=0,
                    end_char=len(text),
                    token_count=len(text.split()),
                    chunking_strategy="counting",
                )
            ]

    name = "counting-test-strategy"
    CHUNKING_STRATEGIES.register(name, lambda **kwargs: _CountingChunker())
    try:
        chunker: ChunkingStrategy = create_chunking_strategy(name)
        chunks = chunker.chunk("alpha beta gamma", ChunkConfig())
    finally:
        CHUNKING_STRATEGIES.unregister(name)

    assert isinstance(chunker, _CountingChunker)
    assert [chunk.text for chunk in chunks] == ["alpha beta gamma"]
    assert chunks[0].chunking_strategy == "counting"


def test_router_delegates_to_registry() -> None:
    """Router resolves strategy via the registry, not an internal switch."""
    captured: dict[str, Any] = {}

    class _Sentinel:
        def chunk(self, text: str, config: ChunkConfig) -> List[PreparedChunk]:
            captured["text"] = text
            captured["config"] = config
            return []

    from rag_core.documents.chunking.builtins import _build_markdown_chunker

    CHUNKING_STRATEGIES.unregister(MARKDOWN_CHUNKING_STRATEGY)
    CHUNKING_STRATEGIES.register(
        MARKDOWN_CHUNKING_STRATEGY, lambda **kwargs: _Sentinel()
    )
    try:
        result = chunk_text(
            "hello world", config=ChunkConfig(strategy=MARKDOWN_CHUNKING_STRATEGY)
        )
    finally:
        CHUNKING_STRATEGIES.unregister(MARKDOWN_CHUNKING_STRATEGY)
        CHUNKING_STRATEGIES.register(
            MARKDOWN_CHUNKING_STRATEGY, _build_markdown_chunker
        )

    assert result == []
    assert captured["text"] == "hello world"


@pytest.mark.parametrize(
    ("opening_fence", "nonmatching_fence", "closing_fence"),
    [
        ("```markdown", "~~~", "```"),
        ("~~~~markdown", "```", "~~~~"),
    ],
    ids=["backticks", "tildes"],
)
def test_markdown_chunker_ignores_headings_inside_matching_fences(
    opening_fence: str,
    nonmatching_fence: str,
    closing_fence: str,
) -> None:
    text = (
        "# Root\n\n"
        f"{opening_fence}\n"
        "## Page 99\n"
        f"{nonmatching_fence}\n"
        "## Still fenced\n"
        f"{closing_fence}\n\n"
        "## After fence\n\n"
        "AFTER-FENCE-EVIDENCE"
    )

    chunks = MarkdownChunker().chunk(
        text,
        ChunkConfig(max_chars=1_000, overlap=0),
    )

    assert len(chunks) == 2
    fenced, after = chunks
    assert "## Page 99" in fenced.text
    assert "## Still fenced" in fenced.text
    assert fenced.metadata["section_path"] == "Root"
    assert fenced.metadata["section_title"] == "Root"
    assert after.metadata["section_path"] == "Root > After fence"
    assert after.metadata["section_title"] == "After fence"
    assert all(
        chunk.start_char is not None
        and chunk.end_char is not None
        and text[chunk.start_char : chunk.end_char] == chunk.text
        for chunk in chunks
    )


def test_markdown_chunker_preserves_four_space_indented_atx_code() -> None:
    text = (
        "# Root\n\n"
        "    ## Page 99\n"
        "    INDENTED-CODE-EVIDENCE\n\n"
        "   ## After code\n\n"
        "AFTER-INDENTED-CODE-EVIDENCE"
    )

    chunks = MarkdownChunker().chunk(
        text,
        ChunkConfig(max_chars=1_000, overlap=0),
    )

    assert len(chunks) == 2
    indented, after = chunks
    assert "## Page 99" in indented.text
    assert indented.metadata["section_path"] == "Root"
    assert indented.metadata["section_title"] == "Root"
    assert after.metadata["section_path"] == "Root > After code"
    assert after.metadata["section_title"] == "After code"
    assert all(
        chunk.start_char is not None
        and chunk.end_char is not None
        and text[chunk.start_char : chunk.end_char] == chunk.text
        for chunk in chunks
    )
