from __future__ import annotations

import asyncio

from rag_core.search.providers.memory_store import InMemoryVectorStore
from rag_core.search.providers.qdrant_store import QdrantVectorStore
from rag_core.search.providers.turbopuffer_store import TurboPufferVectorStore
from rag_core.search.providers.vector_store_capabilities import (
    MEMORY_VECTOR_STORE_CAPABILITY_SPEC,
    QDRANT_VECTOR_STORE_CAPABILITY_SPEC,
    TURBOPUFFER_VECTOR_STORE_CAPABILITY_SPEC,
)
from rag_core.search.types import QueryPlanCapabilities, StoreCapabilities


def test_query_plan_capabilities_default_to_no_optional_plan_support() -> None:
    capabilities = StoreCapabilities(
        per_point_delete=False,
        document_record_lookup=False,
    )

    assert capabilities.query_plan == QueryPlanCapabilities()
    assert capabilities.query_plan.hybrid is False


def test_memory_store_declares_spec_query_plan_support() -> None:
    capabilities = InMemoryVectorStore().capabilities

    assert capabilities.query_plan == MEMORY_VECTOR_STORE_CAPABILITY_SPEC.query_plan
    assert capabilities.query_plan.hybrid is True
    assert capabilities.query_plan.mmr is False
    assert capabilities.query_plan.boost is False


def test_qdrant_store_declares_spec_query_plan_support() -> None:
    store = QdrantVectorStore(
        url=None,
        api_key=None,
        collection_name="docs",
        location=":memory:",
        dense_dimensions=4,
    )
    try:
        assert store.capabilities.query_plan == QDRANT_VECTOR_STORE_CAPABILITY_SPEC.query_plan
        assert store.capabilities.query_plan.hybrid is True
        assert store.capabilities.query_plan.boost is True
    finally:
        asyncio.run(store.close())


def test_turbopuffer_store_declares_spec_query_plan_support() -> None:
    store = TurboPufferVectorStore(
        namespace="docs",
        dense_dimensions=4,
        namespace_client=object(),
    )

    assert store.capabilities.query_plan == TURBOPUFFER_VECTOR_STORE_CAPABILITY_SPEC.query_plan
    assert store.capabilities.query_plan.hybrid is True
    assert store.capabilities.query_plan.mmr is False
    assert store.capabilities.query_plan.boost is False
