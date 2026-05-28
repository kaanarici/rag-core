from __future__ import annotations

from pathlib import Path

from rag_core.config import (
    BUILTIN_CHUNKING_STRATEGIES,
    CHUNKING_STRATEGY_AUTO,
    CODE_CHUNKING_STRATEGY,
    CONTENT_CHUNKER_CHUNKING_STRATEGY,
    MARKDOWN_CHUNKING_STRATEGY,
    PRECHUNKED_CHUNKING_STRATEGY,
    SEMANTIC_CHUNKING_STRATEGY,
)

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
            "src/rag_core/core_prepare_chunks.py",
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

    owner = sources["src/rag_core/config/chunking_config.py"]
    assert owner.count('CHUNKING_STRATEGY_AUTO = "auto"') == 1
    assert owner.count('MARKDOWN_CHUNKING_STRATEGY = "markdown"') == 1
    assert owner.count('SEMANTIC_CHUNKING_STRATEGY = "semantic"') == 1
    assert owner.count('CODE_CHUNKING_STRATEGY = "code"') == 1
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
