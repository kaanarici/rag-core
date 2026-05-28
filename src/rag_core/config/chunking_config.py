from __future__ import annotations

from dataclasses import dataclass

CHUNKING_STRATEGY_AUTO = "auto"
MARKDOWN_CHUNKING_STRATEGY = "markdown"
SEMANTIC_CHUNKING_STRATEGY = "semantic"
CODE_CHUNKING_STRATEGY = "code"
PRECHUNKED_CHUNKING_STRATEGY = "prechunked"
CONTENT_CHUNKER_CHUNKING_STRATEGY = "content_chunker"
BUILTIN_CHUNKING_STRATEGIES = (
    MARKDOWN_CHUNKING_STRATEGY,
    SEMANTIC_CHUNKING_STRATEGY,
    CODE_CHUNKING_STRATEGY,
)


@dataclass(frozen=True)
class ChunkingConfig:
    """Empty by design; the chunking router selects strategy from content type."""
