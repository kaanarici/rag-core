"""Route text through the configured chunking strategy."""

from __future__ import annotations

from typing import Awaitable, Callable, List, Optional, cast

from rag_core.config.chunking_config import (
    CHUNKING_STRATEGY_AUTO,
    CODE_CHUNKING_STRATEGY,
    MARKDOWN_CHUNKING_STRATEGY,
    SEMANTIC_CHUNKING_STRATEGY,
)
from rag_core.core_models import PreparedChunk

from .code_detection import (
    detect_code_language,
    is_code_content as is_code_content,
)
from .protocol import AsyncChunkingStrategy, ChunkConfig
from .registry import create_chunking_strategy


EmbedFn = Callable[[List[str]], Awaitable[List[List[float]]]]


def _resolve_strategy(
    text: str,
    config: Optional[ChunkConfig],
    *,
    mime_type: Optional[str],
    filename: Optional[str],
) -> tuple[ChunkConfig, str]:
    """Resolve the chunk config and concrete strategy (expanding ``auto``)."""
    config = config or ChunkConfig()
    strategy = config.strategy
    if strategy == CHUNKING_STRATEGY_AUTO:
        strategy = _detect_strategy(text, mime_type=mime_type, filename=filename)
    return config, strategy


def chunk_text(
    text: str,
    *,
    config: Optional[ChunkConfig] = None,
    mime_type: Optional[str] = None,
    filename: Optional[str] = None,
) -> List[PreparedChunk]:
    """Chunk text with the resolved strategy."""
    config, strategy = _resolve_strategy(
        text, config, mime_type=mime_type, filename=filename
    )

    if strategy == CODE_CHUNKING_STRATEGY:
        language = detect_code_language(mime_type=mime_type, filename=filename)
        chunker = create_chunking_strategy(CODE_CHUNKING_STRATEGY, language=language)
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
    config, strategy = _resolve_strategy(
        text, config, mime_type=mime_type, filename=filename
    )

    if strategy == SEMANTIC_CHUNKING_STRATEGY:
        chunker = cast(
            AsyncChunkingStrategy,
            create_chunking_strategy(SEMANTIC_CHUNKING_STRATEGY, embed_fn=embed_fn),
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
        return CODE_CHUNKING_STRATEGY

    return MARKDOWN_CHUNKING_STRATEGY
