from __future__ import annotations

import ast
import asyncio
from dataclasses import replace

import pytest

from rag_core import Engine
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
from rag_core.core_models import PreparedChunk
from rag_core.documents.chunking.semantic import SemanticChunker
from rag_core.search.providers.memory_store import InMemoryVectorStore
from tests.support import FakeEmbeddingProvider, FakeSparseEmbedder, make_test_config
from tests.support.source_graph import import_graph, iter_package_sources

CHUNKING_CONFIG_OWNER = "rag_core.config.chunking_config"

# Roots that participate in chunking-strategy naming: the config owner plus the
# layers that route by strategy. Single-ownership is asserted across this union
# so the constants cannot be re-declared closer to a call site.
STRATEGY_ROOTS = (
    "src/rag_core/config",
    "src/rag_core/core_models.py",
    "src/rag_core/documents/prepare_chunks.py",
    "src/rag_core/documents/chunking",
    "src/rag_core/search/indexer_points.py",
    "src/rag_core/search/indexer_prepare.py",
)


def _modules_defining(name: str, *roots: str) -> set[str]:
    """Dotted modules under ``roots`` with a top-level def/class/assign of ``name``."""
    owners: set[str] = set()
    for _rel, dotted, source in iter_package_sources(*roots):
        for node in ast.iter_child_nodes(ast.parse(source)):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                if node.name == name:
                    owners.add(dotted)
            elif isinstance(node, ast.Assign):
                if any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                    owners.add(dotted)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id == name:
                    owners.add(dotted)
    return owners


def test_chunking_strategy_names_have_single_config_owner() -> None:
    # Public strategy values are a contract; assert them directly.
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

    # Each strategy name is declared exactly once, in the config owner module --
    # no strategy-routing layer redeclares it closer to a call site.
    for constant in (
        "CHUNKING_STRATEGY_AUTO",
        "MARKDOWN_CHUNKING_STRATEGY",
        "SEMANTIC_CHUNKING_STRATEGY",
        "CODE_CHUNKING_STRATEGY",
        "PRECHUNKED_CHUNKING_STRATEGY",
        "CONTENT_CHUNKER_CHUNKING_STRATEGY",
    ):
        assert _modules_defining(constant, *STRATEGY_ROOTS) == {CHUNKING_CONFIG_OWNER}

    # The chunking layer reaches the names through the owner rather than
    # re-inlining raw strategy strings.
    graph = import_graph("src/rag_core/documents/chunking", "src/rag_core/config")
    importers = {
        dotted
        for dotted, imports in graph.items()
        if any(
            imported == CHUNKING_CONFIG_OWNER
            or imported.startswith(f"{CHUNKING_CONFIG_OWNER}.")
            for imported in imports
        )
    }
    assert importers != set()


def test_public_chunking_config_controls_facade_prepare_path() -> None:
    async def go() -> list[str]:
        core = Engine(
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


def test_public_prepare_semantic_strategy_uses_async_chunker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def fake_chunk_async(
        self: SemanticChunker,
        text: str,
        config: object,
    ) -> list[PreparedChunk]:
        del self, config
        calls.append(text)
        return [
            PreparedChunk(
                chunk_index=0,
                text=text,
                embedding_text=text,
                word_count=len(text.split()),
                start_char=0,
                end_char=len(text),
                token_count=1,
                chunking_strategy=SEMANTIC_CHUNKING_STRATEGY,
                metadata={"chunking_strategy": SEMANTIC_CHUNKING_STRATEGY},
            )
        ]

    monkeypatch.setattr(SemanticChunker, "chunk_async", fake_chunk_async)

    async def go() -> list[str]:
        core = Engine(
            replace(
                make_test_config(embedding_dimensions=4),
                chunking=ChunkingConfig(
                    strategy=SEMANTIC_CHUNKING_STRATEGY,
                    max_chars=80,
                    overlap=0,
                ),
            ),
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=InMemoryVectorStore(),
        )
        prepared = await core.prepare_bytes(
            file_bytes=b"Alpha topic. Beta topic. Gamma topic.",
            filename="guide.md",
            mime_type="text/markdown",
        )
        return [chunk.chunking_strategy for chunk in prepared.chunks]

    strategies = asyncio.run(go())

    assert calls == ["Alpha topic. Beta topic. Gamma topic."]
    assert strategies == [SEMANTIC_CHUNKING_STRATEGY]


def test_chunking_config_validates_public_shape() -> None:
    with pytest.raises(ValueError, match="strategy"):
        ChunkingConfig(strategy="made-up")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="overlap"):
        ChunkingConfig(max_chars=10, overlap=10)
