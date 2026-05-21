"""Route text through the configured chunking strategy."""

from __future__ import annotations

from rag_core.core_models import PreparedChunk
from typing import Awaitable, Callable, List, Optional, cast

from .code_detection import (
    detect_code_language,
    is_code_content as is_code_content,
)
from .protocol import AsyncChunkingStrategy, ChunkConfig
from .registry import create_chunking_strategy


EmbedFn = Callable[[List[str]], Awaitable[List[List[float]]]]


def chunk_text(
    text: str,
    *,
    config: Optional[ChunkConfig] = None,
    mime_type: Optional[str] = None,
    filename: Optional[str] = None,
) -> List[PreparedChunk]:
    """Chunk text with the resolved strategy."""
    if config is None:
        config = ChunkConfig()

    strategy = config.strategy
    if strategy == "auto":
        strategy = _detect_strategy(text, mime_type=mime_type, filename=filename)

    if strategy == "code":
        language = detect_code_language(mime_type=mime_type, filename=filename)
        chunker = create_chunking_strategy("code", language=language)
    else:
        chunker = create_chunking_strategy(strategy)
    return chunker.chunk(text, config)


async def chunk_text_async(
    text: str,
    *,
    config: Optional[ChunkConfig] = None,
    mime_type: Optional[str] = None,
    filename: Optional[str] = None,
    embed_fn: Optional[EmbedFn] = None,
) -> List[PreparedChunk]:
    """Chunk text asynchronously when the strategy requires embedding calls."""
    if config is None:
        config = ChunkConfig()

    strategy = config.strategy
    if strategy == "auto":
        strategy = _detect_strategy(text, mime_type=mime_type, filename=filename)

    if strategy == "semantic":
        chunker = cast(
            AsyncChunkingStrategy,
            create_chunking_strategy("semantic", embed_fn=embed_fn),
        )
        return await chunker.chunk_async(text, config)

    return chunk_text(
        text,
        config=config,
        mime_type=mime_type,
        filename=filename,
    )


def _detect_strategy(
    text: str,
    *,
    mime_type: Optional[str] = None,
    filename: Optional[str] = None,
) -> str:
    """Auto-detect a chunking strategy from content hints."""
    if is_code_content(
        mime_type=mime_type,
        filename=filename,
        text=text,
    ):
        return "code"

    return "markdown"
