from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from rag_core import RAGCore
from rag_core.cli_doctor import _planned_runtime_payload
from rag_core.cli_doctor_output import emit_doctor
from rag_core.config import EmbeddingConfig
from rag_core.core_models import RAGCoreConfig
from rag_core.search import (
    DEFAULT_SEARCH_PROFILE,
    QUERY_PLAN_PRESETS,
    SEARCH_PROFILES,
    describe_retrieval_profiles,
    search_profile,
)
from rag_core.search.providers.memory_store import InMemoryVectorStore

from tests.support import FakeEmbeddingProvider, FakeSparseEmbedder, make_test_config


def _mapping(value: object) -> dict[str, Any]:
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def test_retrieval_profile_diagnostics_match_profile_builders() -> None:
    payload = describe_retrieval_profiles()

    assert payload["default_search_profile"] == DEFAULT_SEARCH_PROFILE
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


def test_doctor_payload_exposes_retrieval_profile_diagnostics() -> None:
    payload = _planned_runtime_payload(
        RAGCoreConfig(
            embedding=EmbeddingConfig(model="text-embedding-3-small", dimensions=4),
        )
    )

    retrieval = _mapping(payload["retrieval"])
    profiles = _mapping(retrieval["search_profiles"])
    presets = _mapping(retrieval["query_plan_presets"])
    assert retrieval["default_search_profile"] == "balanced"
    assert _mapping(profiles["fast"])["preset"] == "dense_only"
    assert _mapping(profiles["lexical"])["preset"] == "sparse_only"
    assert _mapping(presets["hybrid_dbsf"])["fusion"] == "dbsf"


def test_describe_runtime_exposes_retrieval_profile_diagnostics() -> None:
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

    retrieval = _mapping(payload["retrieval"])
    profiles = _mapping(retrieval["search_profiles"])
    assert retrieval["default_search_profile"] == "balanced"
    assert _mapping(profiles["coverage"])["quality"] == "broad"


def test_doctor_human_output_summarizes_retrieval_profiles(
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
                "requested": "none",
                "effective": "none",
                "fallback_reason": None,
            },
            "qdrant": {"url": None, "location": ":memory:"},
            "vector_store": {},
            "providers": {},
            "retrieval": describe_retrieval_profiles(),
        },
        as_json=False,
        fix=False,
    )

    output = capsys.readouterr().out
    assert "Retrieval Profiles:" in output
    assert (
        "* balanced: preset=hybrid_rrf latency=medium quality=balanced "
        "use=general-purpose hybrid retrieval"
    ) in output
    assert (
        "- diverse: preset=hybrid_with_mmr latency=higher quality=diverse "
        "use=hybrid retrieval with diversity reranking"
    ) in output
