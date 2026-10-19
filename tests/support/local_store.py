"""Real Qdrant :memory: adapter with small test dimensions; no store emulation."""

from __future__ import annotations
from rag_core.search.policy import DEFAULT_POLICY, VectorStorePolicy
from rag_core.search.providers.qdrant_store import QdrantVectorStore


class LocalQdrantStore(QdrantVectorStore):
    def __init__(
        self, policy: VectorStorePolicy = DEFAULT_POLICY, *, dense_dimensions: int = 4
    ) -> None:
        super().__init__(
            url=None,
            api_key=None,
            location=":memory:",
            collection_name="test_chunks",
            dense_dimensions=dense_dimensions,
            sparse_enabled=True,
            policy=policy,
        )
