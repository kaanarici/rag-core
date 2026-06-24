"""Built-in ChunkingStrategy registrations.

Imported lazily by :data:`CHUNKING_STRATEGIES` on first lookup so chunker
modules (which may pull in optional deps for semantic/code chunking) only
load when the registry is actually used.
"""

from __future__ import annotations

from typing import Any

from rag_core.config.chunking_config import (
    CODE_CHUNKING_STRATEGY,
    MARKDOWN_CHUNKING_STRATEGY,
    SEMANTIC_CHUNKING_STRATEGY,
)

from .code import CodeChunker
from .markdown import MarkdownChunker
from .registry import CHUNKING_STRATEGIES
from .semantic import SemanticChunker


def _build_markdown_chunker(**_: Any) -> MarkdownChunker:
    return MarkdownChunker()


def _build_semantic_chunker(**kwargs: Any) -> SemanticChunker:
    return SemanticChunker(**kwargs)


def _build_code_chunker(**kwargs: Any) -> CodeChunker:
    return CodeChunker(**kwargs)


CHUNKING_STRATEGIES.register(MARKDOWN_CHUNKING_STRATEGY, _build_markdown_chunker)
CHUNKING_STRATEGIES.register(SEMANTIC_CHUNKING_STRATEGY, _build_semantic_chunker)
CHUNKING_STRATEGIES.register(CODE_CHUNKING_STRATEGY, _build_code_chunker)
