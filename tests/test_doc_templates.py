"""Product docs and agent templates stay separated."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

TEMPLATE_PATHS = (
    "docs/templates/README.md",
    "docs/templates/AGENTS.md",
    "docs/templates/CONTEXT.md",
    "docs/templates/MISSION.md",
    "docs/templates/ROUTING.md",
)

GITIGNORE_MARKERS = (
    "/docs/plans/",
    "/docs/research/",
    "/docs/AGENTS.md",
    "/docs/CONTEXT.md",
)


def test_agent_templates_exist() -> None:
    missing = [rel for rel in TEMPLATE_PATHS if not (REPO_ROOT / rel).is_file()]
    assert not missing, missing


def test_gitignore_blocks_agent_pollution() -> None:
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    for marker in GITIGNORE_MARKERS:
        assert marker in gitignore, marker


def test_setup_agent_docs_script_exists() -> None:
    path = REPO_ROOT / "scripts/setup_agent_docs.sh"
    assert path.is_file()
    assert path.stat().st_mode & 0o111
