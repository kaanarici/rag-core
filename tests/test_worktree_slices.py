from __future__ import annotations

import json

import pytest

from scripts.worktree_slices import (
    ChangedPath,
    classify_path,
    ensure_no_uncategorized,
    group_changes,
    parse_name_status,
    parse_porcelain,
    render_markdown,
    select_slice,
    serialize_groups,
    validate_slice_key,
)

pytestmark = [pytest.mark.meta]


def test_parse_porcelain_keeps_status_and_rename_target() -> None:
    changes = parse_porcelain(
        [
            "M  README.md",
            "A  scripts/worktree_slices.py",
            "R  src/rag_core/search/searcher.py -> src/rag_core/search/pipeline_runner.py",
        ]
    )

    assert changes == [
        ChangedPath(status="M", path="README.md"),
        ChangedPath(status="A", path="scripts/worktree_slices.py"),
        ChangedPath(status="R", path="src/rag_core/search/pipeline_runner.py"),
    ]


def test_parse_name_status_handles_cached_renames() -> None:
    changes = parse_name_status(
        [
            "M\tREADME.md",
            "R100\tsrc/rag_core/search/searcher.py\tsrc/rag_core/search/pipeline_runner.py",
        ]
    )

    assert changes == [
        ChangedPath(status="M", path="README.md"),
        ChangedPath(status="R", path="src/rag_core/search/pipeline_runner.py"),
    ]


def test_classify_paths_into_reviewable_landing_slices() -> None:
    assert classify_path("scripts/landing_check.sh") == "validation-tooling"
    assert classify_path("docs/templates/README.md") == "repo-packaging-hygiene"
    assert classify_path("docs/self-host/openapi.yaml") == "runtime-self-host"
    assert classify_path("src/rag_core/local_search/eval_runner.py") == "public-api-cli-config"
    assert classify_path("src/rag_core/local_search/runner.py") == "public-api-cli-config"
    assert classify_path("src/rag_core/search/query_plan.py") == "search-retrieval-pipeline"
    assert classify_path("src/rag_core/search/providers/qdrant_store.py") == "providers-integrations"
    assert classify_path("tests/test_pdf_converter_log_sanitization.py") == "documents-parsing"
    assert classify_path("tests/support/fakes.py") == "tests-fixtures-support"


def test_render_markdown_includes_counts_files_and_validation_commands() -> None:
    grouped = group_changes(
        [
            ChangedPath(status="M", path="README.md"),
            ChangedPath(status="A", path="scripts/worktree_slices.py"),
            ChangedPath(status="M", path="src/rag_core/search/query_plan.py"),
        ]
    )

    report = render_markdown(grouped, mode="worktree", max_files=10)

    assert "# rag-core worktree slices" in report
    assert "Mode: worktree" in report
    assert "Changed paths: 3" in report
    assert "## Validation Tooling (1 paths; A:1)" in report
    assert "## Search And Retrieval Pipeline (1 paths; M:1)" in report
    assert "- `./scripts/landing_check.sh --quick`" in report
    assert "- `M` src/rag_core/search/query_plan.py" in report


def test_json_summary_is_deterministic_and_machine_readable() -> None:
    grouped = group_changes(
        [
            ChangedPath(status="M", path="src/rag_core/events/search_events.py"),
            ChangedPath(status="D", path="docs/templates/README.md"),
        ]
    )

    payload = serialize_groups(grouped, mode="staged")
    encoded = json.dumps(payload, sort_keys=True)

    assert payload["mode"] == "staged"
    assert payload["changed_paths"] == 2
    assert "events-traces-evals" in encoded
    assert "repo-packaging-hygiene" in encoded


def test_select_slice_filters_report_to_one_known_slice() -> None:
    grouped = group_changes(
        [
            ChangedPath(status="M", path="scripts/worktree_slices.py"),
            ChangedPath(status="M", path="src/rag_core/search/query_plan.py"),
        ]
    )

    selected = select_slice(grouped, "validation-tooling")
    report = render_markdown(selected, mode="staged", max_files=10)

    assert "## Validation Tooling (1 paths; M:1)" in report
    assert "Search And Retrieval Pipeline" not in report


def test_unknown_slice_key_exits_with_known_slice_hint() -> None:
    with pytest.raises(SystemExit) as exc_info:
        validate_slice_key("unknown-slice")

    assert "unknown slice 'unknown-slice'" in str(exc_info.value)
    assert "validation-tooling" in str(exc_info.value)


def test_fail_on_uncategorized_requires_explicit_routing() -> None:
    grouped = group_changes([ChangedPath(status="M", path="unowned/path.txt")])

    with pytest.raises(SystemExit) as exc_info:
        ensure_no_uncategorized(grouped)

    assert "uncategorized changed paths need routing" in str(exc_info.value)
    assert "unowned/path.txt" in str(exc_info.value)


def test_fail_on_uncategorized_accepts_fully_classified_groups() -> None:
    grouped = group_changes([ChangedPath(status="M", path="scripts/worktree_slices.py")])

    ensure_no_uncategorized(grouped)
