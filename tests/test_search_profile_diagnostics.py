from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from rag_core import RAGCore
from rag_core.cli_doctor import _planned_core_payload
from rag_core.cli_doctor_output import emit_doctor
from rag_core.config import DEFAULT_RERANKER_PROVIDER, EmbeddingConfig
from rag_core.core_models import RAGCoreConfig
from rag_core.search import (
    DEFAULT_SEARCH_PROFILE,
    QUERY_PLAN_PRESETS,
    SEARCH_PROFILES,
    describe_query_plan,
    describe_search_profile_catalog,
    search_profile,
)
from rag_core.search.providers.memory_store import InMemoryVectorStore
from rag_core.search.types import QueryPlanCapabilities

from tests.support import FakeEmbeddingProvider, FakeSparseEmbedder, make_test_config


def _mapping(value: object) -> dict[str, Any]:
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def test_search_profile_diagnostics_match_profile_builders() -> None:
    payload = describe_search_profile_catalog()

    assert payload["default_search_profile"] == DEFAULT_SEARCH_PROFILE
    assert payload["default_search_profile_scope"] == "catalog_profile_when_supported"
    assert payload["default_query_plan_behavior"] == "capability_aware"
    assert "capability-aware defaults may fall back" in str(
        payload["default_search_profile_note"]
    )
    profiles = _mapping(payload["search_profiles"])
    presets = _mapping(payload["query_plan_presets"])
    assert tuple(profiles) == SEARCH_PROFILES
    assert tuple(presets) == QUERY_PLAN_PRESETS

    for name, profile_value in profiles.items():
        profile = _mapping(profile_value)
        preset_name = profile["preset"]
        assert preset_name in presets
        built = search_profile(name, limit=7)
        assert built.final_limit == 7

    balanced = _mapping(profiles["balanced"])
    assert balanced["default"] is True
    assert balanced["preset"] == "hybrid_rrf"
    diverse_preset = _mapping(presets["hybrid_with_mmr"])
    assert diverse_preset["rerank"] == "mmr"
    assert diverse_preset["channels"] == ["dense", "sparse"]


def test_search_profile_catalog_can_describe_effective_default_query_plan() -> None:
    dense_only = describe_search_profile_catalog(
        capabilities=QueryPlanCapabilities(dense=True),
        result_limit=7,
    )
    dense_plan = _mapping(dense_only["effective_default_query_plan"])

    assert dense_plan["search_profile"] is None
    assert dense_plan["channels"] == ["dense"]
    assert dense_plan["fusion"] is None
    assert dense_plan["final_limit"] == 7

    hybrid = describe_search_profile_catalog(
        capabilities=QueryPlanCapabilities(dense=True, sparse=True, hybrid_rrf=True),
        result_limit=9,
    )
    hybrid_plan = _mapping(hybrid["effective_default_query_plan"])

    assert hybrid_plan["search_profile"] == "balanced"
    assert hybrid_plan["channels"] == ["dense", "sparse"]
    assert hybrid_plan["fusion"] == "rrf"
    assert hybrid_plan["final_limit"] == 9


def test_search_facade_exports_query_plan_description_helpers() -> None:
    plan = search_profile("fast", limit=6)
    payload = describe_query_plan(plan)

    assert payload is not None
    assert payload["search_profile"] == "fast"
    assert payload["channels"] == ["dense"]
    assert payload["final_limit"] == 6
    assert "search_profiles" in describe_search_profile_catalog()


def test_doctor_payload_exposes_search_profile_diagnostics() -> None:
    payload = _planned_core_payload(
        RAGCoreConfig(
            embedding=EmbeddingConfig(model="text-embedding-3-small", dimensions=4),
        )
    )

    assert "retrieval" not in payload
    search = _mapping(payload["search"])
    profiles = _mapping(search["search_profiles"])
    presets = _mapping(search["query_plan_presets"])
    assert search["default_search_profile"] == "balanced"
    assert search["default_search_profile_scope"] == "catalog_profile_when_supported"
    assert search["default_query_plan_behavior"] == "capability_aware"
    assert "effective_default_query_plan" not in search
    assert _mapping(profiles["fast"])["preset"] == "dense_only"
    assert _mapping(profiles["lexical"])["preset"] == "sparse_only"
    assert _mapping(presets["hybrid_dbsf"])["fusion"] == "dbsf"


def test_describe_runtime_exposes_search_profile_diagnostics() -> None:
    core = RAGCore(
        make_test_config(embedding_dimensions=4),
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=InMemoryVectorStore(),
    )
    try:
        payload = core.describe_runtime()
    finally:
        asyncio.run(core.close())

    assert "retrieval" not in payload
    search = _mapping(payload["search"])
    profiles = _mapping(search["search_profiles"])
    default_plan = _mapping(search["effective_default_query_plan"])
    assert search["default_search_profile"] == "balanced"
    assert search["default_search_profile_scope"] == "catalog_profile_when_supported"
    assert default_plan["search_profile"] == "balanced"
    assert default_plan["channels"] == ["dense", "sparse"]
    assert default_plan["fusion"] == "rrf"
    assert _mapping(profiles["coverage"])["quality"] == "broad"


def test_doctor_human_output_summarizes_search_profiles(
    capsys: pytest.CaptureFixture[str],
) -> None:
    emit_doctor(
        {
            "runtime": {},
            "collection_name": "docs",
            "processing_version": "v1",
            "source_processing_versions": {},
            "embedding": {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "dimensions": 1536,
                "batch_size": 128,
            },
            "reranker": {
                "requested": DEFAULT_RERANKER_PROVIDER,
                "effective": DEFAULT_RERANKER_PROVIDER,
                "fallback_reason": None,
            },
            "qdrant": {"url": None, "location": ":memory:"},
            "vector_store": {},
            "providers": {},
            "search": describe_search_profile_catalog(),
        },
        as_json=False,
        fix=False,
    )

    output = capsys.readouterr().out
    assert "Search Profiles:" in output
    assert (
        "* balanced: preset=hybrid_rrf latency=medium quality=balanced "
        "use=general-purpose hybrid retrieval"
    ) in output
    assert (
        "- diverse: preset=hybrid_with_mmr latency=higher quality=diverse "
        "use=hybrid retrieval with diversity reranking"
    ) in output
    assert "default: balanced is the catalog default" in output
