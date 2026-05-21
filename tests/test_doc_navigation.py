"""Guardrails for the layered documentation map agents rely on."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DOC_PATHS = (
    "docs/CONTEXT.md",
    "docs/AGENTS.md",
    "docs/README.md",
    "docs/plans/ROUTING.md",
    "docs/plans/one-repo-retrieval-engine-strategy.md",
    "docs/quickstart.md",
    "docs/expectations.md",
    "scripts/README.md",
)

ROUTING_MUST_REFERENCE = (
    "../CONTEXT.md",
    "./scripts/dx_smoke.sh",
    "archive/",
)


def test_required_navigation_files_exist() -> None:
    missing = [rel for rel in REQUIRED_DOC_PATHS if not (REPO_ROOT / rel).is_file()]
    assert not missing, f"missing navigation files: {missing}"


def test_routing_points_at_context_and_smoke() -> None:
    routing = (REPO_ROOT / "docs/plans/ROUTING.md").read_text(encoding="utf-8")
    for needle in ROUTING_MUST_REFERENCE:
        assert needle in routing, needle
