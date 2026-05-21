"""Chunking strategy protocol and types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

from rag_core.core_models import PreparedChunk


@dataclass(frozen=True)
class ChunkConfig:
    """Configuration for chunking."""

    max_chars: int = 2000
    overlap: int = 200
    strategy: str = "auto"  # "auto", "markdown", "semantic", "code"


class ChunkingStrategy(Protocol):
    """Protocol for chunking strategies."""

    def chunk(self, text: str, config: ChunkConfig) -> List[PreparedChunk]: ...


class AsyncChunkingStrategy(Protocol):
    """Protocol for async chunking strategies."""

    async def chunk_async(self, text: str, config: ChunkConfig) -> List[PreparedChunk]: ...
