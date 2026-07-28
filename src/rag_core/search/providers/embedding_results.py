"""Shared provider result validation for dense embedding adapters."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real
from typing import TypeGuard

from rag_core.search.providers.provider_result_values import safe_provider_value_type


def _is_int_index(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real_value(value: object) -> TypeGuard[Real]:
    return isinstance(value, Real) and not isinstance(value, bool)


def safe_indexed_embedding_vectors(
    *,
    rows: list[tuple[object, object]],
    expected_count: int,
    expected_dimensions: int | None,
    provider_name: str,
) -> list[list[float]]:
    _validate_expected_shape(
        expected_count=expected_count,
        expected_dimensions=expected_dimensions,
    )
    if len(rows) != expected_count:
        raise ValueError(
            "%s returned embedding count mismatch: expected %d vectors, got %d"
            % (provider_name, expected_count, len(rows))
        )
    vectors: list[list[float] | None] = [None] * expected_count
    for result_index, (raw_index, raw_vector) in enumerate(rows):
        if not _is_int_index(raw_index) or not 0 <= raw_index < expected_count:
            raise ValueError(
                "%s returned invalid embedding index at result index %d "
                "(value_type=%s)"
                % (provider_name, result_index, safe_provider_value_type(raw_index))
            )
        if vectors[raw_index] is not None:
            raise ValueError(
                "%s returned duplicate embedding index at result index %d"
                % (provider_name, result_index)
            )
        vectors[raw_index] = _validate_embedding_vector(
            raw_vector,
            expected_dimensions=expected_dimensions,
            provider_name=provider_name,
            result_index=result_index,
        )
    ordered: list[list[float]] = []
    for vector in vectors:
        if vector is None:
            raise ValueError(
                "%s returned embedding count mismatch: expected %d vectors, got %d"
                % (provider_name, expected_count, len(rows))
            )
        ordered.append(vector)
    return ordered


def safe_ordered_embedding_vectors(
    *,
    rows: list[object],
    expected_count: int,
    expected_dimensions: int | None,
    provider_name: str,
) -> list[list[float]]:
    _validate_expected_shape(
        expected_count=expected_count,
        expected_dimensions=expected_dimensions,
    )
    if len(rows) != expected_count:
        raise ValueError(
            "%s returned embedding count mismatch: expected %d vectors, got %d"
            % (provider_name, expected_count, len(rows))
        )
    return [
        _validate_embedding_vector(
            raw_vector,
            expected_dimensions=expected_dimensions,
            provider_name=provider_name,
            result_index=result_index,
        )
        for result_index, raw_vector in enumerate(rows)
    ]


def _validate_expected_shape(
    *,
    expected_count: int,
    expected_dimensions: int | None,
) -> None:
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count < 0
    ):
        raise ValueError("embedding expected_count must be a non-negative integer")
    if expected_dimensions is None:
        return
    if (
        isinstance(expected_dimensions, bool)
        or not isinstance(expected_dimensions, int)
        or expected_dimensions <= 0
    ):
        raise ValueError("embedding expected_dimensions must be a positive integer")


def _validate_embedding_vector(
    raw_vector: object,
    *,
    expected_dimensions: int | None,
    provider_name: str,
    result_index: int,
) -> list[float]:
    if (
        isinstance(raw_vector, (str, bytes, bytearray))
        or not isinstance(raw_vector, Sequence)
    ):
        raise ValueError(
            "%s returned invalid embedding vector at result index %d "
            "(reason=invalid_type value_type=%s)"
            % (provider_name, result_index, safe_provider_value_type(raw_vector))
        )
    if expected_dimensions is not None and len(raw_vector) != expected_dimensions:
        raise ValueError(
            "%s returned embedding dimension mismatch at result index %d: "
            "expected %d dimensions, got %d"
            % (provider_name, result_index, expected_dimensions, len(raw_vector))
        )
    if expected_dimensions is None and len(raw_vector) == 0:
        raise ValueError(
            "%s returned empty embedding vector at result index %d"
            % (provider_name, result_index)
        )
    vector: list[float] = []
    for value_index, raw_value in enumerate(raw_vector):
        if not _is_real_value(raw_value):
            raise ValueError(
                "%s returned invalid embedding value at result index %d "
                "dimension %d (reason=invalid_type value_type=%s)"
                % (
                    provider_name,
                    result_index,
                    value_index,
                    safe_provider_value_type(raw_value),
                )
            )
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(
                "%s returned non-finite embedding value at result index %d "
                "dimension %d (value_type=%s)"
                % (
                    provider_name,
                    result_index,
                    value_index,
                    safe_provider_value_type(raw_value),
                )
            )
        vector.append(value)
    return vector
