from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from rag_core import RAGCore
from rag_core.config import (
    BUILTIN_CHUNKING_STRATEGIES,
    CHUNKING_STRATEGY_AUTO,
    CODE_CHUNKING_STRATEGY,
    CONTENT_CHUNKER_CHUNKING_STRATEGY,
    MARKDOWN_CHUNKING_STRATEGY,
    PRECHUNKED_CHUNKING_STRATEGY,
    PUBLIC_CHUNKING_STRATEGIES,
    SEMANTIC_CHUNKING_STRATEGY,
    ChunkingConfig,
)
from rag_core.search.providers.memory_store import InMemoryVectorStore
from tests.support import FakeEmbeddingProvider, FakeSparseEmbedder, make_test_config

CANONICAL_LAUNCH_GATES = (
    "uv run ruff check .",
    "uv run mypy src tests examples",
    "uv run pytest -q",
    "./scripts/dx_smoke.sh",
    "./scripts/ci_self_host_smoke.sh",
    "uv build",
    "uv run python scripts/check_dist_artifacts.py",
    "uv run python scripts/wheel_smoke.py",
)


def test_chunking_strategy_names_have_single_config_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/config/chunking_config.py",
            "src/rag_core/config/__init__.py",
            "src/rag_core/core_models.py",
            "src/rag_core/_engine/core_prepare_chunks.py",
            "src/rag_core/documents/chunking/protocol.py",
            "src/rag_core/documents/chunking/builtins.py",
            "src/rag_core/documents/chunking/router.py",
            "src/rag_core/documents/chunking/markdown.py",
            "src/rag_core/documents/chunking/code_segments.py",
            "src/rag_core/documents/chunking/semantic.py",
            "src/rag_core/documents/chunking/semantic_chunk_builder.py",
            "src/rag_core/search/indexer_points.py",
            "src/rag_core/search/indexer_texts.py",
        )
    }

    assert CHUNKING_STRATEGY_AUTO == "auto"
    assert MARKDOWN_CHUNKING_STRATEGY == "markdown"
    assert SEMANTIC_CHUNKING_STRATEGY == "semantic"
    assert CODE_CHUNKING_STRATEGY == "code"
    assert PRECHUNKED_CHUNKING_STRATEGY == "prechunked"
    assert CONTENT_CHUNKER_CHUNKING_STRATEGY == "content_chunker"
    assert BUILTIN_CHUNKING_STRATEGIES == (
        MARKDOWN_CHUNKING_STRATEGY,
        SEMANTIC_CHUNKING_STRATEGY,
        CODE_CHUNKING_STRATEGY,
    )
    assert PUBLIC_CHUNKING_STRATEGIES == (
        CHUNKING_STRATEGY_AUTO,
        MARKDOWN_CHUNKING_STRATEGY,
        SEMANTIC_CHUNKING_STRATEGY,
        CODE_CHUNKING_STRATEGY,
    )

    owner = sources["src/rag_core/config/chunking_config.py"]
    assert owner.count('CHUNKING_STRATEGY_AUTO: ChunkingStrategyName = "auto"') == 1
    assert owner.count(
        'MARKDOWN_CHUNKING_STRATEGY: ChunkingStrategyName = "markdown"'
    ) == 1
    assert owner.count(
        'SEMANTIC_CHUNKING_STRATEGY: ChunkingStrategyName = "semantic"'
    ) == 1
    assert owner.count('CODE_CHUNKING_STRATEGY: ChunkingStrategyName = "code"') == 1
    assert owner.count('PRECHUNKED_CHUNKING_STRATEGY = "prechunked"') == 1
    assert owner.count('CONTENT_CHUNKER_CHUNKING_STRATEGY = "content_chunker"') == 1

    consumers = "\n".join(
        source
        for path, source in sources.items()
        if path != "src/rag_core/config/chunking_config.py"
    )
    assert "CHUNKING_STRATEGY_AUTO" in consumers
    assert "MARKDOWN_CHUNKING_STRATEGY" in consumers
    assert "SEMANTIC_CHUNKING_STRATEGY" in consumers
    assert "CODE_CHUNKING_STRATEGY" in consumers
    assert "PRECHUNKED_CHUNKING_STRATEGY" in consumers
    assert "CONTENT_CHUNKER_CHUNKING_STRATEGY" in consumers
    assert 'strategy: str = "auto"' not in consumers
    assert 'chunking_strategy: str = "prechunked"' not in consumers
    assert 'chunking_strategy="markdown"' not in consumers
    assert 'chunking_strategy="semantic"' not in consumers
    assert 'chunking_strategy="code"' not in consumers
    assert 'chunking_strategy=req.chunker_strategy or "prechunked"' not in consumers
    assert (
        '"prechunked" if req.pre_chunked_texts else "content_chunker"' not in consumers
    )
    assert 'strategy_name="semantic"' not in consumers
    assert 'return "markdown"' not in consumers
    assert 'return "semantic"' not in consumers
    assert 'return "code"' not in consumers
    assert 'CHUNKING_STRATEGIES.register("markdown"' not in consumers
    assert 'CHUNKING_STRATEGIES.register("semantic"' not in consumers
    assert 'CHUNKING_STRATEGIES.register("code"' not in consumers


def test_public_chunking_config_controls_facade_prepare_path() -> None:
    async def go() -> list[str]:
        core = RAGCore(
            replace(
                make_test_config(embedding_dimensions=4),
                chunking=ChunkingConfig(strategy="markdown", max_chars=80, overlap=0),
            ),
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=InMemoryVectorStore(),
        )
        prepared = await core.prepare_bytes(
            file_bytes=(
                b"# One\n\n"
                + b"alpha " * 40
                + b"\n\n# Two\n\n"
                + b"beta " * 40
            ),
            filename="guide.md",
            mime_type="text/markdown",
        )
        return [chunk.chunking_strategy for chunk in prepared.chunks]

    strategies = asyncio.run(go())

    assert strategies
    assert set(strategies) == {MARKDOWN_CHUNKING_STRATEGY}


def test_chunking_config_validates_public_shape() -> None:
    with pytest.raises(ValueError, match="strategy"):
        ChunkingConfig(strategy="made-up")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="overlap"):
        ChunkingConfig(max_chars=10, overlap=10)
