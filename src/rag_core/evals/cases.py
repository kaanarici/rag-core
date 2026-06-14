"""Eval case models and JSONL loading."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from rag_core.evals.case_fields import (
    expected_ids_tuple,
    load_error,
    non_negative_int,
    optional_grades,
    optional_non_negative_int,
    optional_positive_int,
    optional_positive_int_field,
    optional_string,
    optional_string_tuple,
    required_string,
    required_string_tuple,
    validate_expected_grades,
    validate_unique_expected_ids,
)


@dataclass(frozen=True, init=False)
class EvalCase:
    """One labelled query for the eval runner.

    ``expected_ids`` is the neutral name for ids the runner treats as relevant.
    Results are matched against ``SearchResult.id`` first when the case names
    exact chunks, then against ``document_id`` or ``document_key`` for
    document-level cases.

    ``expected_grades`` is required only for graded ``ndcg_at_k``. When
    absent, nDCG falls back to binary relevance from ``expected_ids``.
    """

    query: str
    namespace: str
    corpus_ids: tuple[str, ...]
    expected_ids: tuple[str, ...]
    expected_grades: Mapping[str, int] | None = None
    case_id: str | None = None
    expected_context_contains: tuple[str, ...] = ()
    forbidden_context_contains: tuple[str, ...] = ()
    forbidden_private_identifiers: tuple[str, ...] = ()
    expected_citation_count_min: int = 0
    expected_source_count_min: int = 0
    max_context_chars: int | None = None
    max_context_tokens: int | None = None

    def __init__(
        self,
        query: str,
        namespace: str,
        corpus_ids: Sequence[str],
        expected_ids: Sequence[str],
        expected_grades: Mapping[str, int] | None = None,
        case_id: str | None = None,
        expected_context_contains: Sequence[str] = (),
        forbidden_context_contains: Sequence[str] = (),
        forbidden_private_identifiers: Sequence[str] = (),
        expected_citation_count_min: int = 0,
        expected_source_count_min: int = 0,
        max_context_chars: int | None = None,
        max_context_tokens: int | None = None,
    ) -> None:
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "corpus_ids", tuple(corpus_ids))
        object.__setattr__(self, "expected_ids", tuple(expected_ids))
        object.__setattr__(self, "expected_grades", expected_grades)
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(
            self,
            "expected_context_contains",
            tuple(expected_context_contains),
        )
        object.__setattr__(
            self,
            "forbidden_context_contains",
            tuple(forbidden_context_contains),
        )
        object.__setattr__(
            self,
            "forbidden_private_identifiers",
            tuple(forbidden_private_identifiers),
        )
        object.__setattr__(
            self,
            "expected_citation_count_min",
            non_negative_int(
                expected_citation_count_min,
                "expected_citation_count_min",
            ),
        )
        object.__setattr__(
            self,
            "expected_source_count_min",
            non_negative_int(expected_source_count_min, "expected_source_count_min"),
        )
        object.__setattr__(
            self,
            "max_context_chars",
            optional_positive_int(max_context_chars, "max_context_chars"),
        )
        object.__setattr__(
            self,
            "max_context_tokens",
            optional_positive_int(max_context_tokens, "max_context_tokens"),
        )


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
                raise load_error(
                    case_path,
                    line_number,
                    f"duplicate object key {exc.key!r}",
                ) from None
            except json.JSONDecodeError:
                raise load_error(case_path, line_number, "invalid JSON") from None
            case = _load_case_row(row, path=case_path, line_number=line_number)
            if case.case_id is not None:
                first_seen_line = seen_case_ids.get(case.case_id)
                if first_seen_line is not None:
                    raise load_error(
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
        raise load_error(path, line_number, "case must be a JSON object")
    expected_ids = expected_ids_tuple(row, path=path, line_number=line_number)
    validate_unique_expected_ids(
        expected_ids,
        path=path,
        line_number=line_number,
    )
    expected_grades = optional_grades(row, path=path, line_number=line_number)
    validate_expected_grades(
        expected_ids,
        expected_grades,
        path=path,
        line_number=line_number,
    )
    return EvalCase(
        query=required_string(row, "query", path=path, line_number=line_number),
        namespace=required_string(row, "namespace", path=path, line_number=line_number),
        corpus_ids=required_string_tuple(
            row,
            "corpus_ids",
            path=path,
            line_number=line_number,
        ),
        expected_ids=expected_ids,
        expected_grades=expected_grades,
        case_id=optional_string(row, "case_id", path=path, line_number=line_number),
        expected_context_contains=optional_string_tuple(
            row,
            "expected_context_contains",
            path=path,
            line_number=line_number,
        ),
        forbidden_context_contains=optional_string_tuple(
            row,
            "forbidden_context_contains",
            path=path,
            line_number=line_number,
        ),
        forbidden_private_identifiers=optional_string_tuple(
            row,
            "forbidden_private_identifiers",
            path=path,
            line_number=line_number,
        ),
        expected_citation_count_min=optional_non_negative_int(
            row,
            "expected_citation_count_min",
            path=path,
            line_number=line_number,
        ),
        expected_source_count_min=optional_non_negative_int(
            row,
            "expected_source_count_min",
            path=path,
            line_number=line_number,
        ),
        max_context_chars=optional_positive_int_field(
            row,
            "max_context_chars",
            path=path,
            line_number=line_number,
        ),
        max_context_tokens=optional_positive_int_field(
            row,
            "max_context_tokens",
            path=path,
            line_number=line_number,
        ),
    )

__all__ = ["EvalCase", "load_cases"]
