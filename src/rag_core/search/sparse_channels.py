"""The built-in lexical channel; stores may retain older unused vector fields."""
from __future__ import annotations
from typing import TypeVar

PRIMARY_SPARSE_CHANNEL = "bm25"
KNOWN_SPARSE_CHANNELS = frozenset({PRIMARY_SPARSE_CHANNEL})
T = TypeVar("T")

def single_sparse_channel(vector: T) -> dict[str, T]:
    return {PRIMARY_SPARSE_CHANNEL: vector}
