"""Tests for the pluggable sidecar merge strategies."""

from __future__ import annotations

import asyncio

import pytest

from rag_core.search.pipeline import (
    PreferMaxScoreMerge,
    PreferSidecarMerge,
    ScoreBlendMerge,
)

from tests.support import make_search_result


def test_prefer_max_score_merges_dedups_and_keeps_higher_score() -> None:
    """Duplicate ids collapse onto the sidecar text but keep the max score and
    fall back to vector-side richer fields. Metadata unions with sidecar winning."""

    async def _run() -> None:
        vector_only = make_search_result(id="vector-only", score=0.5, text="vector text")
        sidecar_only = make_search_result(
            id="sidecar-only", score=1.0, text="sidecar text", title="Sidecar"
        )
        duplicate_vector = make_search_result(
            id="dup",
            score=0.6,
            text="vector text",
            document_key="/docs/guide.txt",
            content_sha256="hash",
            section_title="Overview",
            metadata={"from": "vector", "extra": "vec"},
        )
        duplicate_sidecar = make_search_result(
            id="dup",
            score=0.0,
            text="sidecar exact",
            metadata={"from": "sidecar", "shared": "side"},
        )

        merged = await PreferMaxScoreMerge().merge(
            [duplicate_vector, vector_only],
            [duplicate_sidecar, sidecar_only],
        )
        assert [r.id for r in merged] == ["sidecar-only", "dup", "vector-only"]
        dup = merged[1]
        assert dup.text == "sidecar exact"
        assert dup.score == 0.6
        assert dup.document_key == "/docs/guide.txt"
        assert dup.content_sha256 == "hash"
        assert dup.section_title == "Overview"
        assert dup.metadata == {"from": "sidecar", "shared": "side", "extra": "vec"}

    asyncio.run(_run())


@pytest.mark.parametrize(
    "vector_ids, sidecar_ids",
    [
        (["a", "b"], []),
        ([], ["a", "b"]),
    ],
)
def test_prefer_max_score_passes_through_when_other_side_empty(
    vector_ids: list[str], sidecar_ids: list[str]
) -> None:
    async def _run() -> None:
        merged = await PreferMaxScoreMerge().merge(
            [make_search_result(id=i) for i in vector_ids],
            [make_search_result(id=i) for i in sidecar_ids],
        )
        assert [r.id for r in merged] == (vector_ids or sidecar_ids)

    asyncio.run(_run())


def test_prefer_max_score_does_not_promote_weak_sidecar_hits_above_vector_hits() -> None:
    async def _run() -> None:
        merged = await PreferMaxScoreMerge().merge(
            [make_search_result(id="vector", score=0.99)],
            [make_search_result(id="sidecar", score=0.36)],
        )

        assert [(result.id, result.score) for result in merged] == [
            ("vector", 0.99),
            ("sidecar", 0.36),
        ]

    asyncio.run(_run())


def test_prefer_sidecar_puts_sidecar_first_and_keeps_unique_vector_results() -> None:
    async def _run() -> None:
        strategy = PreferSidecarMerge()
        vector = [
            make_search_result(id="a", score=0.99),
            make_search_result(id="b", score=0.5),
        ]
        sidecar = [make_search_result(id="b", score=0.1)]
        merged = await strategy.merge(vector, sidecar)
        assert [r.id for r in merged] == ["b", "a"]
        assert merged[0].score == 0.5

        unique = await strategy.merge(
            [make_search_result(id="vec-only", score=0.7)],
            [make_search_result(id="side", score=0.4)],
        )
        assert [r.id for r in unique] == ["vec-only", "side"]

    asyncio.run(_run())


@pytest.mark.parametrize(
    "alpha, expected_dup_score",
    [
        (0.0, 0.9),
        (1.0, 0.4),
        (0.5, 0.65),
    ],
)
def test_score_blend_interpolates_duplicate_scores(
    alpha: float, expected_dup_score: float
) -> None:
    async def _run() -> None:
        merged = await ScoreBlendMerge(alpha=alpha).merge(
            [make_search_result(id="dup", score=0.4)],
            [make_search_result(id="dup", score=0.9)],
        )
        assert merged[0].score == pytest.approx(expected_dup_score)

    asyncio.run(_run())


def test_score_blend_passes_unique_results_through_unchanged() -> None:
    async def _run() -> None:
        merged = await ScoreBlendMerge(alpha=0.5).merge(
            [make_search_result(id="vec", score=0.4)],
            [make_search_result(id="side", score=0.9)],
        )
        assert [r.id for r in merged] == ["side", "vec"]
        assert merged[0].score == pytest.approx(0.9)
        assert merged[1].score == pytest.approx(0.4)

    asyncio.run(_run())


@pytest.mark.parametrize("alpha", [-0.1, 1.5])
def test_score_blend_rejects_alpha_out_of_range(alpha: float) -> None:
    with pytest.raises(ValueError):
        ScoreBlendMerge(alpha=alpha)
