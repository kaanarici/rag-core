from __future__ import annotations

from typing import Final, Literal, TypeAlias

EmbeddingInputType: TypeAlias = Literal["document", "query"]

EMBEDDING_INPUT_DOCUMENT: Final[EmbeddingInputType] = "document"
EMBEDDING_INPUT_QUERY: Final[EmbeddingInputType] = "query"

__all__ = [
    "EMBEDDING_INPUT_DOCUMENT",
    "EMBEDDING_INPUT_QUERY",
    "EmbeddingInputType",
]
