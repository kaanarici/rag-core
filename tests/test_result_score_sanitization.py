from __future__ import annotations

import asyncio
from typing import Any, cast

from rag_core.search.pipeline import (
    PreferMaxScoreMerge,
    PreferSidecarMerge,
    ScoreBlendMerge,
)
from rag_core.search.result_scores import finite_score_or_zero
from rag_core.search.stored_payload import merge_duplicate_result

from tests.support import make_search_result


def test_finite_score_or_zero_drops_malformed_scores() -> None:
    assert finite_score_or_zero(float("nan")) == 0.0
    assert finite_score_or_zero(float("inf")) == 0.0
    assert finite_score_or_zero(True) == 0.0
    assert finite_score_or_zero(False) == 0.0
    assert finite_score_or_zero(cast(Any, object())) == 0.0
    assert finite_score_or_zero(0.75) == 0.75


def test_merge_duplicate_result_sanitizes_scores() -> None:
    preferred = make_search_result(id="dup", score=float("nan"))
    fallback = make_search_result(id="dup", score=0.4)

    assert merge_duplicate_result(preferred, fallback).score == 0.4
    assert (
        merge_duplicate_result(
            preferred,
            make_search_result(id="dup", score=float("inf")),
        ).score
        == 0.0
    )


def test_prefer_max_score_merge_sanitizes_unique_and_duplicate_scores() -> None:
    async def _run() -> None:
        merged = await PreferMaxScoreMerge().merge(
            [
                make_search_result(id="dup", score=0.6),
                make_search_result(id="vector", score=cast(Any, False)),
            ],
            [
                make_search_result(id="dup", score=float("nan")),
                make_search_result(id="sidecar", score=-float("inf")),
            ],
        )

        assert [(result.id, result.score) for result in merged] == [
            ("dup", 0.6),
            ("sidecar", 0.0),
            ("vector", 0.0),
        ]

    asyncio.run(_run())


def test_prefer_sidecar_merge_sanitizes_scores() -> None:
    async def _run() -> None:
        merged = await PreferSidecarMerge().merge(
            [make_search_result(id="vector", score=float("nan"))],
            [make_search_result(id="sidecar", score=float("inf"))],
        )

        assert [(result.id, result.score) for result in merged] == [
            ("sidecar", 0.0),
            ("vector", 0.0),
        ]

    asyncio.run(_run())


def test_score_blend_merge_sanitizes_inputs_and_outputs() -> None:
    async def _run() -> None:
        merged = await ScoreBlendMerge(alpha=0.5).merge(
            [
                make_search_result(id="dup", score=float("nan")),
                make_search_result(id="vector", score=float("inf")),
            ],
            [
                make_search_result(id="dup", score=0.8),
                make_search_result(id="sidecar", score=-float("inf")),
            ],
        )

        assert [(result.id, result.score) for result in merged] == [
            ("dup", 0.4),
            ("sidecar", 0.0),
            ("vector", 0.0),
        ]

    asyncio.run(_run())
