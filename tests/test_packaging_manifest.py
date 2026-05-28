from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_dist_artifacts import FORBIDDEN_SDIST_SUFFIXES, REQUIRED_SDIST_SUFFIXES

pytestmark = [pytest.mark.meta]


def test_sdist_manifest_includes_readme_link_targets() -> None:
    manifest = Path("MANIFEST.in").read_text(encoding="utf-8")

    required_lines = (
        "include compose.yaml",
        "include Dockerfile",
        "include .env.example",
        "include docs/embed.md",
        "include docs/expectations.md",
        "include docs/parsing/formats.md",
        "include docs/providers.md",
        "include docs/quickstart.md",
        "include docs/self-host.md",
        "include docs/self-host/openapi.yaml",
        "include docs/stability.md",
        "recursive-include examples *.jsonl *.md *.py *.ts",
        "recursive-include scripts *.py *.sh",
        "prune tests",
    )
    for line in required_lines:
        assert line in manifest


def test_public_docs_and_examples_are_present_for_packaging() -> None:
    for target in REQUIRED_SDIST_SUFFIXES:
        assert Path(target).is_file()


def test_reviewability_script_is_required_sdist_artifact() -> None:
    assert "scripts/worktree_slices.py" in REQUIRED_SDIST_SUFFIXES


def test_sdist_manifest_excludes_local_agent_docs() -> None:
    manifest = Path("MANIFEST.in").read_text(encoding="utf-8")

    assert "recursive-include docs" not in manifest
    for suffix in FORBIDDEN_SDIST_SUFFIXES:
        assert f"include {suffix}" not in manifest


def test_dist_artifact_checker_forbidden_artifacts_are_not_public_artifacts() -> None:
    overlap = REQUIRED_SDIST_SUFFIXES & FORBIDDEN_SDIST_SUFFIXES

    assert not overlap
    assert any(suffix.startswith("docs/templates/") for suffix in FORBIDDEN_SDIST_SUFFIXES)
    assert any(suffix.startswith("dev/") for suffix in FORBIDDEN_SDIST_SUFFIXES)
    assert any(
        suffix.startswith("scripts/") and suffix.endswith(".sh")
        for suffix in FORBIDDEN_SDIST_SUFFIXES
    )
