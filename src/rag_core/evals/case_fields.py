"""Field validation helpers for JSONL eval case loading."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


def required_string(
    row: dict[str, object],
    field: str,
    *,
    path: Path,
    line_number: int,
) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise load_error(path, line_number, f"{field} must be a non-empty string")
    return value.strip()


def optional_string(
    row: dict[str, object],
    field: str,
    *,
    path: Path,
    line_number: int,
) -> str | None:
    value = row.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise load_error(path, line_number, f"{field} must be a non-empty string when set")
    return value.strip()


def required_string_tuple(
    row: dict[str, object],
    field: str,
    *,
    path: Path,
    line_number: int,
) -> tuple[str, ...]:
    value = row.get(field)
    if not isinstance(value, list) or not value:
        raise load_error(path, line_number, f"{field} must be a non-empty string array")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise load_error(path, line_number, f"{field} must contain only non-empty strings")
        items.append(item.strip())
    return tuple(items)


def optional_string_tuple(
    row: dict[str, object],
    field: str,
    *,
    path: Path,
    line_number: int,
) -> tuple[str, ...]:
    if field not in row:
        return ()
    return required_string_tuple(row, field, path=path, line_number=line_number)


def optional_non_negative_int(
    row: dict[str, object],
    field: str,
    *,
    path: Path,
    line_number: int,
) -> int:
    value = row.get(field)
    if value is None:
        return 0
    try:
        return non_negative_int(value, field)
    except ValueError as exc:
        raise load_error(path, line_number, str(exc)) from None


def optional_positive_int_field(
    row: dict[str, object],
    field: str,
    *,
    path: Path,
    line_number: int,
) -> int | None:
    value = row.get(field)
    try:
        return optional_positive_int(value, field)
    except ValueError as exc:
        raise load_error(path, line_number, str(exc)) from None


def non_negative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def optional_positive_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def expected_ids_tuple(
    row: dict[str, object],
    *,
    path: Path,
    line_number: int,
) -> tuple[str, ...]:
    has_expected_ids = "expected_ids" in row
    has_expected_chunk_ids = "expected_chunk_ids" in row
    if has_expected_ids and has_expected_chunk_ids:
        raise load_error(
            path,
            line_number,
            "use expected_ids or expected_chunk_ids, not both",
        )
    field = "expected_ids" if has_expected_ids else "expected_chunk_ids"
    return required_string_tuple(row, field, path=path, line_number=line_number)


def optional_grades(
    row: dict[str, object],
    *,
    path: Path,
    line_number: int,
) -> Mapping[str, int] | None:
    value = row.get("expected_grades")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise load_error(path, line_number, "expected_grades must be an object")
    grades: dict[str, int] = {}
    for item_id, grade in value.items():
        if not isinstance(item_id, str) or not item_id.strip():
            raise load_error(
                path,
                line_number,
                "expected_grades keys must be non-empty strings",
            )
        stripped_id = item_id.strip()
        if stripped_id in grades:
            raise load_error(
                path,
                line_number,
                f"duplicate expected_grades key {stripped_id!r}",
            )
        if isinstance(grade, bool) or not isinstance(grade, int) or grade < 0:
            raise load_error(
                path,
                line_number,
                "expected_grades values must be non-negative integers",
            )
        grades[stripped_id] = grade
    return grades


def validate_unique_expected_ids(
    expected_ids: tuple[str, ...],
    *,
    path: Path,
    line_number: int,
) -> None:
    if len(set(expected_ids)) != len(expected_ids):
        raise load_error(
            path,
            line_number,
            "expected_ids must not contain duplicate ids",
        )


def validate_expected_grades(
    expected_ids: tuple[str, ...],
    expected_grades: Mapping[str, int] | None,
    *,
    path: Path,
    line_number: int,
) -> None:
    if expected_grades is None:
        return
    relevant_ids = set(expected_ids)
    positive_grade_ids = {
        item_id for item_id, grade in expected_grades.items() if grade > 0
    }
    if positive_grade_ids != relevant_ids:
        raise load_error(
            path,
            line_number,
            "expected_grades positive ids must match expected_ids",
        )


def load_error(path: Path, line_number: int, reason: str) -> ValueError:
    return ValueError(f"{path}:{line_number}: {reason}")


__all__ = [
    "expected_ids_tuple",
    "load_error",
    "non_negative_int",
    "optional_grades",
    "optional_non_negative_int",
    "optional_positive_int",
    "optional_positive_int_field",
    "optional_string",
    "optional_string_tuple",
    "required_string",
    "required_string_tuple",
    "validate_expected_grades",
    "validate_unique_expected_ids",
]
