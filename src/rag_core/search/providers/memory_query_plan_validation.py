from __future__ import annotations

from rag_core.search.query_plan import (
    FUSION_KIND_RRF,
    PRIMARY_DENSE_QUERY_VECTOR,
    DenseChannel,
    QueryPlan,
    SparseChannel,
    UnsupportedQueryStage,
)
from rag_core.search.sparse_channels import KNOWN_SPARSE_CHANNELS


def validate_memory_query_plan(plan: QueryPlan) -> None:
    if plan.boost is not None:
        raise UnsupportedQueryStage("Boost is not supported by InMemoryVectorStore")
    if plan.rerank is not None:
        raise UnsupportedQueryStage(
            "MMR rerank is not supported by InMemoryVectorStore"
        )
    if plan.fuse is not None and plan.fuse.kind != FUSION_KIND_RRF:
        raise UnsupportedQueryStage(
            f"{plan.fuse.kind} fusion is not supported by InMemoryVectorStore"
        )

    for prefetch in plan.prefetches:
        if prefetch.nested:
            raise UnsupportedQueryStage(
                "Nested prefetch is not supported by InMemoryVectorStore"
            )
        channel = prefetch.channel
        if isinstance(channel, DenseChannel):
            validate_memory_dense_channel(channel)
        elif isinstance(channel, SparseChannel):
            validate_memory_sparse_channel(channel)
        else:
            raise UnsupportedQueryStage(
                f"Unknown channel type: {type(channel).__name__}"
            )


def validate_memory_dense_channel(channel: DenseChannel) -> None:
    if channel.vector_field or channel.using_query_vector != PRIMARY_DENSE_QUERY_VECTOR:
        raise UnsupportedQueryStage(
            "InMemoryVectorStore supports only the primary dense query vector"
        )


def validate_memory_sparse_channel(channel: SparseChannel) -> None:
    if channel.vector_field not in KNOWN_SPARSE_CHANNELS:
        raise UnsupportedQueryStage(
            f"SparseChannel({channel.vector_field!r}) is not supported by InMemoryVectorStore"
        )


__all__ = [
    "validate_memory_dense_channel",
    "validate_memory_query_plan",
    "validate_memory_sparse_channel",
]
