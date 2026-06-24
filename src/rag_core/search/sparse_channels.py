from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

PRIMARY_SPARSE_CHANNEL = "bm25"
SECONDARY_SPARSE_CHANNEL = "splade"
KNOWN_SPARSE_CHANNELS: frozenset[str] = frozenset(
    {PRIMARY_SPARSE_CHANNEL, SECONDARY_SPARSE_CHANNEL}
)

T = TypeVar("T")


def merge_sparse_channels(
    primary: T,
    extra: Mapping[str, T] | None,
) -> dict[str, T]:
    merged: dict[str, T] = {PRIMARY_SPARSE_CHANNEL: primary}
    if not extra:
        return merged
    for name, vector in extra.items():
        if name:
            merged[str(name)] = vector
    return merged


def primary_sparse_channel(
    channels: Mapping[str, T],
    *,
    missing_message: str,
) -> T:
    primary = channels.get(PRIMARY_SPARSE_CHANNEL)
    if primary is not None:
        return primary
    if channels:
        return next(iter(channels.values()))
    raise ValueError(missing_message)


def single_sparse_channel(vector: T) -> dict[str, T]:
    return {PRIMARY_SPARSE_CHANNEL: vector}
