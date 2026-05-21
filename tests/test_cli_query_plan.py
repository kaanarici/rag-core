"""Tests for the CLI query-plan presets and higher-level search profiles.

Each named recipe must map to a stable plan shape. Validation lives at the
factory level so any addition or rename is visible in one place.
"""

from __future__ import annotations

import pytest

from rag_core.cli import _build_parser
from rag_core.search import (
    QUERY_PLAN_PRESETS,
    SEARCH_PROFILES,
    query_plan_preset,
    search_profile,
)
from rag_core.search.query_plan import (
    DenseChannel,
    Mmr,
    SparseChannel,
)


def test_hybrid_rrf_preset_has_dense_plus_sparse_with_rrf() -> None:
    plan = query_plan_preset("hybrid_rrf", limit=10)

    assert len(plan.prefetches) == 2
    assert isinstance(plan.prefetches[0].channel, DenseChannel)
    assert isinstance(plan.prefetches[1].channel, SparseChannel)
    assert plan.fuse is not None
    assert plan.fuse.kind == "rrf"
    assert plan.rerank is None
    assert plan.final_limit == 10


def test_dense_only_preset_drops_sparse_and_fusion() -> None:
    plan = query_plan_preset("dense_only", limit=5)

    assert len(plan.prefetches) == 1
    assert isinstance(plan.prefetches[0].channel, DenseChannel)
    assert plan.fuse is None
    assert plan.rerank is None
    assert plan.final_limit == 5


def test_sparse_only_preset_targets_bm25_channel() -> None:
    plan = query_plan_preset("sparse_only", limit=8)

    assert len(plan.prefetches) == 1
    sparse_channel = plan.prefetches[0].channel
    assert isinstance(sparse_channel, SparseChannel)
    assert sparse_channel.vector_field == "bm25"
    assert plan.fuse is None
    assert plan.rerank is None


def test_hybrid_dbsf_preset_uses_dbsf_fusion() -> None:
    plan = query_plan_preset("hybrid_dbsf", limit=20)

    assert len(plan.prefetches) == 2
    assert plan.fuse is not None
    assert plan.fuse.kind == "dbsf"


def test_hybrid_with_mmr_preset_reranks_widened_candidate_pool() -> None:
    plan = query_plan_preset("hybrid_with_mmr", limit=12)

    assert len(plan.prefetches) == 2
    assert plan.fuse is not None
    assert plan.fuse.kind == "rrf"
    assert isinstance(plan.rerank, Mmr)
    assert plan.rerank.limit == 48
    assert plan.final_limit == 12
    assert 0.0 < plan.rerank.diversity < 1.0


def test_unknown_preset_raises_with_valid_set_listed() -> None:
    with pytest.raises(ValueError) as exc_info:
        query_plan_preset("nope", limit=10)

    message = str(exc_info.value)
    assert "nope" in message
    for name in QUERY_PLAN_PRESETS:
        assert name in message


@pytest.mark.parametrize(
    ("profile", "preset"),
    (
        ("balanced", "hybrid_rrf"),
        ("fast", "dense_only"),
        ("lexical", "sparse_only"),
        ("coverage", "hybrid_dbsf"),
        ("diverse", "hybrid_with_mmr"),
    ),
)
def test_search_profile_maps_to_query_plan_preset(profile: str, preset: str) -> None:
    profile_plan = search_profile(profile, limit=9)
    preset_plan = query_plan_preset(preset, limit=9)
    assert profile_plan.prefetches == preset_plan.prefetches
    assert profile_plan.fuse == preset_plan.fuse
    assert profile_plan.rerank == preset_plan.rerank
    assert profile_plan.boost == preset_plan.boost
    assert profile_plan.final_limit == preset_plan.final_limit
    assert profile_plan.search_profile == profile
    assert preset_plan.search_profile is None


def test_unknown_search_profile_raises_with_valid_set_listed() -> None:
    with pytest.raises(ValueError) as exc_info:
        search_profile("nope", limit=10)

    message = str(exc_info.value)
    assert "nope" in message
    for name in SEARCH_PROFILES:
        assert name in message


def test_query_subparser_accepts_query_plan_preset_flag() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "search",
            "billing policy",
            "--namespace",
            "acme",
            "--corpus-id",
            "help",
            "--qdrant-url",
            "http://localhost:6333",
            "--query-plan-preset",
            "hybrid_with_mmr",
        ]
    )

    assert args.command == "search"
    assert args.query_plan_preset == "hybrid_with_mmr"


def test_query_subparser_accepts_search_profile_flag() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "search",
            "billing policy",
            "--namespace",
            "acme",
            "--corpus-id",
            "help",
            "--qdrant-url",
            "http://localhost:6333",
            "--search-profile",
            "diverse",
        ]
    )

    assert args.command == "search"
    assert args.search_profile == "diverse"


def test_search_command_uses_raw_search_output() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "search",
            "billing policy",
            "--namespace",
            "acme",
            "--corpus-id",
            "help",
            "--qdrant-url",
            "http://localhost:6333",
        ]
    )

    assert args.command == "search"
    assert args.context_json is False


def test_search_command_rejects_context_json_alias() -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "search",
                "billing policy",
                "--namespace",
                "acme",
                "--corpus-id",
                "help",
                "--context-json",
            ]
        )


def test_retrieve_context_command_defaults_to_context_payload() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "retrieve-context",
            "billing policy",
            "--namespace",
            "acme",
            "--corpus-id",
            "help",
            "--qdrant-url",
            "http://localhost:6333",
        ]
    )

    assert args.command == "retrieve-context"
    assert args.context_json is True
    assert args.json is False


def test_query_subparser_default_preset_is_none() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "search",
            "billing policy",
            "--namespace",
            "acme",
            "--corpus-id",
            "help",
            "--qdrant-url",
            "http://localhost:6333",
        ]
    )

    assert args.query_plan_preset is None
    assert args.search_profile is None


def test_query_subparser_rejects_unknown_preset_name() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "search",
                "billing policy",
                "--namespace",
                "acme",
                "--corpus-id",
                "help",
                "--qdrant-url",
                "http://localhost:6333",
                "--query-plan-preset",
                "made-up-preset",
            ]
        )


def test_query_subparser_rejects_unknown_search_profile_name() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "search",
                "billing policy",
                "--namespace",
                "acme",
                "--corpus-id",
                "help",
                "--qdrant-url",
                "http://localhost:6333",
                "--search-profile",
                "made-up-profile",
            ]
        )




def test_known_presets_match_factory_advertised_set() -> None:
    expected = {
        "hybrid_rrf",
        "dense_only",
        "sparse_only",
        "hybrid_dbsf",
        "hybrid_with_mmr",
    }
    assert set(QUERY_PLAN_PRESETS) == expected


def test_known_search_profiles_match_factory_advertised_set() -> None:
    expected = {
        "balanced",
        "fast",
        "lexical",
        "coverage",
        "diverse",
    }
    assert set(SEARCH_PROFILES) == expected
