"""Built-in ChunkingStrategy registrations.

Imported lazily by :data:`CHUNKING_STRATEGIES` on first lookup so chunker
modules (which may pull in optional deps for semantic/code chunking) only
load when the registry is actually used.
"""

from __future__ import annotations

from typing import Any

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


CHUNKING_STRATEGIES.register("markdown", _build_markdown_chunker)
CHUNKING_STRATEGIES.register("semantic", _build_semantic_chunker)
CHUNKING_STRATEGIES.register("code", _build_code_chunker)
