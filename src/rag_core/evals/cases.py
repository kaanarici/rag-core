"""Eval case models and JSONL loading."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalCase:
    """One labelled query for the eval runner.

    ``expected_chunk_ids`` lists ids the runner treats as relevant. Results
    are matched against ``SearchResult.id`` first when the fixture names exact
    chunks, then against ``document_id`` for document-level fixtures.

    ``expected_grades`` is required only for graded ``ndcg_at_k``. When
    absent, nDCG falls back to binary relevance from ``expected_chunk_ids``.
    """

    query: str
    namespace: str
    corpus_ids: tuple[str, ...]
    expected_chunk_ids: tuple[str, ...]
    expected_grades: Mapping[str, int] | None = None
    case_id: str | None = None


class _DuplicateObjectKeyError(ValueError):
    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.key = key


def load_cases(path: Path) -> list[EvalCase]:
    """Load JSONL eval cases (one JSON object per non-empty line)."""
    cases: list[EvalCase] = []
    case_path = Path(path)
    if case_path.exists() and not case_path.is_file():
        raise ValueError(f"{case_path}: cases path must be a JSONL file")
    seen_case_ids: dict[str, int] = {}
    with case_path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line, object_pairs_hook=_reject_duplicate_object_keys)
            except _DuplicateObjectKeyError as exc:
                raise _load_error(
                    case_path,
                    line_number,
                    f"duplicate object key {exc.key!r}",
                ) from None
            except json.JSONDecodeError:
                raise _load_error(case_path, line_number, "invalid JSON") from None
            case = _load_case_row(row, path=case_path, line_number=line_number)
            if case.case_id is not None:
                first_seen_line = seen_case_ids.get(case.case_id)
                if first_seen_line is not None:
                    raise _load_error(
                        case_path,
                        line_number,
                        f"duplicate case_id {case.case_id!r} (first seen at line {first_seen_line})",
                    )
                seen_case_ids[case.case_id] = line_number
            cases.append(case)
    if not cases:
        raise ValueError(f"{case_path}: no eval cases found")
    return cases


def _reject_duplicate_object_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    row: dict[str, object] = {}
    for key, value in pairs:
        if key in row:
            raise _DuplicateObjectKeyError(key)
        row[key] = value
    return row


def _load_case_row(row: object, *, path: Path, line_number: int) -> EvalCase:
    if not isinstance(row, dict):
        raise _load_error(path, line_number, "case must be a JSON object")
    expected_chunk_ids = _required_string_tuple(
        row,
        "expected_chunk_ids",
        path=path,
        line_number=line_number,
    )
    _validate_unique_expected_chunk_ids(
        expected_chunk_ids,
        path=path,
        line_number=line_number,
    )
    expected_grades = _optional_grades(row, path=path, line_number=line_number)
    _validate_expected_grades(
        expected_chunk_ids,
        expected_grades,
        path=path,
        line_number=line_number,
    )
    return EvalCase(
        query=_required_string(row, "query", path=path, line_number=line_number),
        namespace=_required_string(row, "namespace", path=path, line_number=line_number),
        corpus_ids=_required_string_tuple(
            row,
            "corpus_ids",
            path=path,
            line_number=line_number,
        ),
        expected_chunk_ids=expected_chunk_ids,
        expected_grades=expected_grades,
        case_id=_optional_string(row, "case_id", path=path, line_number=line_number),
    )


def _required_string(
    row: dict[str, object],
    field: str,
    *,
    path: Path,
    line_number: int,
) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise _load_error(path, line_number, f"{field} must be a non-empty string")
    return value.strip()


def _optional_string(
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
        raise _load_error(path, line_number, f"{field} must be a non-empty string when set")
    return value.strip()


def _required_string_tuple(
    row: dict[str, object],
    field: str,
    *,
    path: Path,
    line_number: int,
) -> tuple[str, ...]:
    value = row.get(field)
    if not isinstance(value, list) or not value:
        raise _load_error(path, line_number, f"{field} must be a non-empty string array")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise _load_error(path, line_number, f"{field} must contain only non-empty strings")
        items.append(item.strip())
    return tuple(items)


def _optional_grades(
    row: dict[str, object],
    *,
    path: Path,
    line_number: int,
) -> Mapping[str, int] | None:
    value = row.get("expected_grades")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise _load_error(path, line_number, "expected_grades must be an object")
    grades: dict[str, int] = {}
    for item_id, grade in value.items():
        if not isinstance(item_id, str) or not item_id.strip():
            raise _load_error(
                path,
                line_number,
                "expected_grades keys must be non-empty strings",
            )
        stripped_id = item_id.strip()
        if stripped_id in grades:
            raise _load_error(
                path,
                line_number,
                f"duplicate expected_grades key {stripped_id!r}",
            )
        if isinstance(grade, bool) or not isinstance(grade, int) or grade < 0:
            raise _load_error(
                path,
                line_number,
                "expected_grades values must be non-negative integers",
            )
        grades[stripped_id] = grade
    return grades


def _validate_unique_expected_chunk_ids(
    expected_chunk_ids: tuple[str, ...],
    *,
    path: Path,
    line_number: int,
) -> None:
    if len(set(expected_chunk_ids)) != len(expected_chunk_ids):
        raise _load_error(
            path,
            line_number,
            "expected_chunk_ids must not contain duplicate ids",
        )


def _validate_expected_grades(
    expected_chunk_ids: tuple[str, ...],
    expected_grades: Mapping[str, int] | None,
    *,
    path: Path,
    line_number: int,
) -> None:
    if expected_grades is None:
        return
    expected_ids = set(expected_chunk_ids)
    positive_grade_ids = {
        item_id for item_id, grade in expected_grades.items() if grade > 0
    }
    if positive_grade_ids != expected_ids:
        raise _load_error(
            path,
            line_number,
            "expected_grades positive ids must match expected_chunk_ids",
        )


def _load_error(path: Path, line_number: int, reason: str) -> ValueError:
    return ValueError(f"{path}:{line_number}: {reason}")


__all__ = ["EvalCase", "load_cases"]
