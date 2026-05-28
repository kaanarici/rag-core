"""Shared retrieval channel labels."""

from __future__ import annotations

from typing import Final, Literal

RetrievalChannel = Literal["dense", "sparse"]

DENSE_RETRIEVAL_CHANNEL: Final[RetrievalChannel] = "dense"
SPARSE_RETRIEVAL_CHANNEL: Final[RetrievalChannel] = "sparse"
RETRIEVAL_CHANNELS: Final[tuple[RetrievalChannel, ...]] = (
    DENSE_RETRIEVAL_CHANNEL,
    SPARSE_RETRIEVAL_CHANNEL,
)

__all__ = [
    "DENSE_RETRIEVAL_CHANNEL",
    "RETRIEVAL_CHANNELS",
    "RetrievalChannel",
    "SPARSE_RETRIEVAL_CHANNEL",
]
