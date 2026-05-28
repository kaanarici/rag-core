from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path

REQUIRED_WHEEL_MEMBERS = frozenset(
    {
        "rag_core/py.typed",
        "rag_core/__init__.pyi",
        "rag_core/events/__init__.pyi",
        "rag_core/integrations/__init__.pyi",
        "rag_core/search/__init__.pyi",
        "rag_core/search/providers/__init__.pyi",
    }
)

REQUIRED_SDIST_SUFFIXES = frozenset(
    {
        "README.md",
        "compose.yaml",
        "Dockerfile",
        ".env.example",
        "docs/parsing/formats.md",
        "docs/providers.md",
        "docs/embed.md",
        "docs/expectations.md",
        "docs/quickstart.md",
        "docs/self-host.md",
        "docs/stability.md",
        "docs/self-host/openapi.yaml",
        "examples/configured_retrieval.py",
        "examples/embedded_service.py",
        "examples/minimal_app.py",
        "examples/search_endpoint.py",
        "examples/retrieval_eval.py",
        "examples/source_ingest.py",
        "examples/vercel_ai_sdk_search_tool.ts",
        "examples/demo_corpus/corpus_lifecycle.md",
        "scripts/ci_self_host_smoke.sh",
        "scripts/dx_smoke.sh",
        "scripts/landing_check.sh",
        "scripts/self_host_smoke.sh",
        "scripts/validate_provider_fixtures.sh",
        "scripts/verify_vercel_ai_sdk_example.sh",
        "scripts/wheel_smoke.py",
        "scripts/worktree_slices.py",
    }
)

FORBIDDEN_SDIST_SUFFIXES = frozenset(
    {
        "AGENTS.md",
        "CLAUDE.md",
        "CONTEXT.md",
        "MISSION.md",
        "docs/AGENTS.md",
        "docs/CONTEXT.md",
        "docs/templates/AGENTS.md",
        "docs/templates/CONTEXT.md",
        "docs/templates/MISSION.md",
        "docs/templates/README.md",
        "docs/templates/ROUTING.md",
        "docs/plans/ROUTING.md",
        "dev/DESIGN.md",
        "dev/PUBLIC_DOCS_PLAN.md",
        "dev/REBRAND.md",
        "dev/project_identity.local.toml.example",
        "dev/project_identity.toml",
        "scripts/brand_check.sh",
        "scripts/local_rebrand.sh",
        "scripts/setup_agent_docs.sh",
    }
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check built rag-core wheel and sdist public artifacts."
    )
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    args = parser.parse_args()

    dist_dir = args.dist_dir
    wheel = _single(dist_dir, "*.whl")
    sdist = _single(dist_dir, "*.tar.gz")

    _check_wheel(wheel)
    _check_sdist(sdist)


def _single(dist_dir: Path, pattern: str) -> Path:
    matches = sorted(dist_dir.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one {pattern} artifact in {dist_dir}, found {len(matches)}"
        )
    return matches[0]


def _check_wheel(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        members = set(archive.namelist())
    missing = sorted(REQUIRED_WHEEL_MEMBERS - members)
    if missing:
        raise SystemExit(f"wheel is missing required typed artifacts: {missing}")


def _check_sdist(path: Path) -> None:
    with tarfile.open(path, "r:gz") as archive:
        suffixes = {"/".join(member.name.split("/")[1:]) for member in archive.getmembers()}
    missing = sorted(REQUIRED_SDIST_SUFFIXES - suffixes)
    if missing:
        raise SystemExit(f"sdist is missing required public artifacts: {missing}")
    forbidden = sorted(FORBIDDEN_SDIST_SUFFIXES & suffixes)
    if forbidden:
        raise SystemExit(f"sdist includes local-only artifacts: {forbidden}")


if __name__ == "__main__":
    main()
