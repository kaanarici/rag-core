from __future__ import annotations

import asyncio

import pytest

from rag_core.search.planning import validate_query_plan_capabilities
from rag_core.search.query_plan import (
    Boost,
    DenseChannel,
    Prefetch,
    QueryPlan,
    UnsupportedQueryStage,
)
import rag_core.search.providers as provider_exports
from rag_core.search.providers.memory_store import InMemoryVectorStore
from rag_core.search.providers.qdrant_store import QdrantVectorStore
from rag_core.search.providers.turbopuffer_store import TurboPufferVectorStore
from rag_core.search.types import QueryPlanCapabilities, StoreCapabilities


def test_query_plan_capabilities_default_to_no_optional_plan_support() -> None:
    capabilities = StoreCapabilities(
        per_point_delete=False,
        document_record_lookup=False,
    )

    assert capabilities.query_plan == QueryPlanCapabilities()
    assert capabilities.query_plan.hybrid is False
    assert provider_exports.QueryPlanCapabilities is QueryPlanCapabilities


def test_memory_store_declares_basic_query_plan_support() -> None:
    capabilities = InMemoryVectorStore().capabilities

    assert capabilities.query_plan == QueryPlanCapabilities(
        dense=True,
        sparse=True,
        hybrid_rrf=True,
    )
    assert capabilities.query_plan.hybrid is True
    assert capabilities.query_plan.mmr is False
    assert capabilities.query_plan.boost is False


def test_qdrant_store_declares_translator_query_plan_support() -> None:
    store = QdrantVectorStore(
        url=None,
        api_key=None,
        collection_name="docs",
        location=":memory:",
        dense_dimensions=4,
    )
    try:
        assert store.capabilities.query_plan == QueryPlanCapabilities(
            dense=True,
            sparse=True,
            hybrid_rrf=True,
            hybrid_dbsf=True,
            hybrid_weighted_rrf=True,
            mmr=True,
            boost=True,
            nested_prefetch=True,
        )
        assert store.capabilities.query_plan.hybrid is True
        assert store.capabilities.query_plan.boost is True
    finally:
        asyncio.run(store.close())


def test_turbopuffer_store_declares_dense_only_query_plan_support() -> None:
    store = TurboPufferVectorStore(
        namespace="docs",
        dense_dimensions=4,
        namespace_client=object(),
    )

    assert store.capabilities.query_plan == QueryPlanCapabilities(dense=True)
    assert store.capabilities.query_plan.sparse is False
    assert store.capabilities.query_plan.hybrid is False
    assert store.capabilities.query_plan.mmr is False
    assert store.capabilities.query_plan.boost is False


def test_turbopuffer_query_plan_capabilities_reject_boost() -> None:
    store = TurboPufferVectorStore(
        namespace="docs",
        dense_dimensions=4,
        namespace_client=object(),
    )
    plan = QueryPlan(
        prefetches=(Prefetch(channel=DenseChannel(), limit=20),),
        boost=Boost(kind="linear_decay", field="freshness"),
        final_limit=10,
    )

    with pytest.raises(UnsupportedQueryStage, match="does not support boost query plans"):
        validate_query_plan_capabilities(
            plan,
            capabilities=store.capabilities.query_plan,
            backend="turbopuffer",
        )
