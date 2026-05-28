"""Qdrant channel translation helpers for typed query plans."""

from __future__ import annotations

from typing import Any

from qdrant_client import models as rest

from rag_core.search.query_plan import (
    DenseChannel,
    SparseChannel,
    UnsupportedQueryStage,
)
from rag_core.search.request_models import SearchQuery

from .qdrant_shared import _DENSE_VECTOR_NAME


def query_for_qdrant_channel(channel: object, query: SearchQuery) -> Any:
    if isinstance(channel, DenseChannel):
        return query.dense_vector
    if isinstance(channel, SparseChannel):
        sparse = query.all_sparse_vectors().get(channel.using_query_vector)
        if sparse is None:
            raise UnsupportedQueryStage(
                f"SparseChannel({channel.using_query_vector!r}) has no matching sparse query vector"
            )
        return rest.SparseVector(indices=sparse.indices, values=sparse.values)
    raise UnsupportedQueryStage(f"Unknown channel type: {type(channel).__name__}")


def using_for_qdrant_channel(channel: object) -> str:
    if isinstance(channel, DenseChannel):
        return _DENSE_VECTOR_NAME if channel.vector_field == "" else channel.vector_field
    if isinstance(channel, SparseChannel):
        return channel.vector_field
    raise UnsupportedQueryStage(f"Unknown channel type: {type(channel).__name__}")


def ensure_qdrant_sparse_channel_supported(
    channel: object,
    available_sparse_names: frozenset[str] | set[str],
) -> None:
    if isinstance(channel, SparseChannel):
        if channel.vector_field not in available_sparse_names:
            available = ", ".join(sorted(available_sparse_names)) or "none"
            raise UnsupportedQueryStage(
                f"SparseChannel({channel.vector_field!r}) is not available in this "
                f"Qdrant collection; available sparse channels: {available}"
            )
        return
    if isinstance(channel, DenseChannel):
        return
    raise UnsupportedQueryStage(f"Unknown channel type: {type(channel).__name__}")
