"""Unit tests for the eval metric pure functions."""

from __future__ import annotations

import math

import pytest

from rag_core.evals import mrr, ndcg_at_k, recall_at_k


@pytest.mark.parametrize(
    ("retrieved", "relevant", "k", "expected"),
    (
        (["a", "b", "c"], ["a"], 5, 1.0),
        (["a", "b", "c", "d", "e", "target"], ["target"], 5, 0.0),
        # 1 of 2 relevant items inside top-3
        (["a", "x", "b", "y"], ["a", "y"], 3, 0.5),
        # Duplicates do not double-count.
        (["a", "a", "b"], ["a"], 5, 1.0),
        ([], ["a"], 5, 0.0),
        (["a", "b"], [], 5, 0.0),
        (["a"], ["a"], 0, 0.0),
        (["a"], ["a"], -1, 0.0),
    ),
    ids=(
        "perfect-top",
        "outside-window",
        "partial",
        "duplicates-ignored",
        "empty-retrieved",
        "empty-relevant",
        "zero-k",
        "negative-k",
    ),
)
def test_recall_at_k(
    retrieved: list[str], relevant: list[str], k: int, expected: float
) -> None:
    assert recall_at_k(retrieved, relevant, k) == expected


@pytest.mark.parametrize(
    ("retrieved", "relevant", "expected"),
    (
        (["a", "b", "c"], ["a"], 1.0),
        (["x", "y", "a"], ["a"], 1 / 3),
        (["x", "y", "z"], ["a"], 0.0),
        ([], ["a"], 0.0),
        (["a"], [], 0.0),
        # First relevant hit determines rank.
        (["x", "a", "b"], ["a", "b"], 0.5),
    ),
    ids=(
        "first-position",
        "third-position",
        "no-match",
        "empty-retrieved",
        "empty-relevant",
        "uses-first-relevant",
    ),
)
def test_mrr(retrieved: list[str], relevant: list[str], expected: float) -> None:
    assert math.isclose(mrr(retrieved, relevant), expected)


def test_ndcg_at_k_perfect_order_is_one() -> None:
    grades = {"a": 3, "b": 2, "c": 1}
    assert math.isclose(ndcg_at_k(["a", "b", "c"], grades, 3), 1.0)


def test_ndcg_at_k_reversed_order_is_lower() -> None:
    grades = {"a": 3, "b": 2, "c": 1}
    perfect = ndcg_at_k(["a", "b", "c"], grades, 3)
    reversed_score = ndcg_at_k(["c", "b", "a"], grades, 3)
    assert reversed_score < perfect
    assert 0.0 < reversed_score < 1.0


def test_ndcg_at_k_truncates_at_k() -> None:
    grades = {"a": 1, "b": 1, "c": 1}
    # k=2: only first two retrieved contribute, IDCG also truncates to k=2
    assert math.isclose(ndcg_at_k(["a", "b", "c"], grades, 2), 1.0)


def test_ndcg_at_k_repeated_relevant_id_only_gains_once() -> None:
    grades = {"doc-a": 3}
    assert math.isclose(ndcg_at_k(["doc-a", "doc-a"], grades, 10), 1.0)


def test_ndcg_at_k_duplicate_hits_still_occupy_rank_positions() -> None:
    grades = {"a": 3, "b": 2}
    expected = 7.0 / (7.0 + (3.0 / math.log2(3)))

    assert math.isclose(ndcg_at_k(["a", "a", "b"], grades, 2), expected)


@pytest.mark.parametrize(
    ("retrieved", "grades", "k"),
    (
        (["x", "y", "z"], {"a": 3}, 3),
        ([], {"a": 1}, 3),
        (["a"], {"a": 0}, 3),
        (["a"], {"a": 1}, 0),
    ),
    ids=("unrelated-results", "empty-retrieved", "no-positive-grades", "zero-k"),
)
def test_ndcg_at_k_zero_cases(
    retrieved: list[str], grades: dict[str, int], k: int
) -> None:
    assert ndcg_at_k(retrieved, grades, k) == 0.0
