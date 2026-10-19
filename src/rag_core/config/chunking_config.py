from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ChunkingStrategyName = Literal["auto", "markdown"]

CHUNKING_STRATEGY_AUTO: ChunkingStrategyName = "auto"
MARKDOWN_CHUNKING_STRATEGY: ChunkingStrategyName = "markdown"
PRECHUNKED_CHUNKING_STRATEGY = "prechunked"
XLSX_CHUNKING_STRATEGY = "xlsx"
BUILTIN_CHUNKING_STRATEGIES = (MARKDOWN_CHUNKING_STRATEGY,)
PUBLIC_CHUNKING_STRATEGIES = (CHUNKING_STRATEGY_AUTO, *BUILTIN_CHUNKING_STRATEGIES)


@dataclass(frozen=True)
class ChunkingConfig:
    """Public chunking controls used by ``Engine`` prepare and ingest paths."""

    strategy: ChunkingStrategyName = CHUNKING_STRATEGY_AUTO
    max_chars: int = 2000
    overlap: int = 200

    def __post_init__(self) -> None:
        if self.strategy not in PUBLIC_CHUNKING_STRATEGIES:
            raise ValueError(
                "ChunkingConfig.strategy must be one of: "
                + ", ".join(PUBLIC_CHUNKING_STRATEGIES)
            )
        if isinstance(self.max_chars, bool) or not isinstance(self.max_chars, int):
            raise ValueError("ChunkingConfig.max_chars must be a positive integer")
        if self.max_chars <= 0:
            raise ValueError("ChunkingConfig.max_chars must be a positive integer")
        if isinstance(self.overlap, bool) or not isinstance(self.overlap, int):
            raise ValueError("ChunkingConfig.overlap must be a non-negative integer")
        if self.overlap < 0:
            raise ValueError("ChunkingConfig.overlap must be a non-negative integer")
        if self.overlap >= self.max_chars:
            raise ValueError("ChunkingConfig.overlap must be smaller than max_chars")
