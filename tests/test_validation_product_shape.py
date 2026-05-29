from __future__ import annotations

import ast
from pathlib import Path


CANONICAL_LAUNCH_GATES = (
    "uv sync --group dev",
    "uv run ruff check .",
    "uv run mypy src tests examples",
    "uv run pytest -q",
    "./scripts/dx_smoke.sh",
    "./scripts/verify_vercel_ai_sdk_example.sh",
    "./scripts/ci_self_host_smoke.sh",
    "uv build",
    "uv run python scripts/check_dist_artifacts.py",
    "uv run python scripts/wheel_smoke.py",
)
LOCAL_LANDING_CHECK = "./scripts/landing_check.sh"
LOCAL_QUICK_LANDING_CHECK = "./scripts/landing_check.sh --quick"
LOCAL_WORKTREE_SLICES = "uv run python scripts/worktree_slices.py --staged"
LOCAL_WORKTREE_SLICES_FAIL = (
    "uv run python scripts/worktree_slices.py --staged --fail-on-uncategorized"
)
FAST_PYTEST_MARKER = (
    "not live and not eval and not eval_harness and not provider_contract "
    "and not integration"
)
FAST_PYTEST_GATE = f'uv run pytest -q -m "{FAST_PYTEST_MARKER}"'

CI_NON_PYTEST_CHECKS = (
    "./scripts/dx_smoke.sh",
    "./scripts/verify_vercel_ai_sdk_example.sh",
    "./scripts/ci_self_host_smoke.sh",
    "uv run ruff check .",
    "uv run mypy src tests examples",
    "uv run python scripts/architecture_pressure.py --json",
    "uv build",
    "uv run python scripts/check_dist_artifacts.py",
    "uv run python scripts/wheel_smoke.py",
)

CI_PYTEST_TIERS = (
    'uv run pytest -q -m "not live and not eval and not eval_harness and not provider_contract and not integration"',
    "uv run pytest -q -m provider_contract",
    "uv run pytest -q -m integration",
    "uv run pytest -q tests/evals/test_retrieval_eval_pr.py",
)


def test_launch_gate_docs_name_the_canonical_validation_commands() -> None:
    root = Path(__file__).resolve().parents[1]
    docs = {
        path: (root / path).read_text(encoding="utf-8")
        for path in ("README.md", "tests/README.md", "scripts/README.md")
    }
    for path, body in docs.items():
        lines = set(body.splitlines())
        for command in CANONICAL_LAUNCH_GATES:
            assert command in lines, f"{path} missing {command}"
        assert LOCAL_LANDING_CHECK in lines, f"{path} missing local wrapper"
        assert LOCAL_QUICK_LANDING_CHECK in lines, f"{path} missing quick wrapper"


def test_landing_check_wraps_canonical_launch_gate_without_new_surface() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "landing_check.sh").read_text(encoding="utf-8")
    scripts_readme = (root / "scripts" / "README.md").read_text(encoding="utf-8")
    workflow = (root / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    for command in CANONICAL_LAUNCH_GATES:
        assert f"run_step {command}" in script
    assert f"FAST_PYTEST_MARKER='{FAST_PYTEST_MARKER}'" in script
    assert 'run_step uv run pytest -q -m "$FAST_PYTEST_MARKER"' in script
    assert "quick landing check passed" in script
    assert "`./scripts/landing_check.sh --quick` while iterating" in scripts_readme
    assert "landing_check.sh` is a local coordination wrapper" in scripts_readme
    assert "run: ./scripts/landing_check.sh" not in workflow
    assert FAST_PYTEST_GATE in scripts_readme
    assert "./scripts/dx_smoke.sh" in script
    assert "uv run rag-core demo" not in script


def test_worktree_slice_report_is_local_reviewability_tooling() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "worktree_slices.py").read_text(encoding="utf-8")
    scripts_readme = (root / "scripts" / "README.md").read_text(encoding="utf-8")
    workflow = (root / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    assert LOCAL_WORKTREE_SLICES in scripts_readme
    assert LOCAL_WORKTREE_SLICES_FAIL in scripts_readme
    assert "uv run python scripts/worktree_slices.py --staged --slice validation-tooling" in scripts_readme
    assert "uv run python scripts/worktree_slices.py --list-slices" in scripts_readme
    assert "Local reviewability report" in scripts_readme
    assert "Groups current git changes into reviewable landing slices" in scripts_readme
    assert '"status", "--porcelain=v1"' in script
    assert '"diff", "--cached", "--name-status"' in script
    assert "--fail-on-uncategorized" in script
    assert "--slice" in script
    assert "run: uv run python scripts/worktree_slices.py" not in workflow


def test_test_readme_names_ci_workflow_non_pytest_checks() -> None:
    root = Path(__file__).resolve().parents[1]
    body = (root / "tests" / "README.md").read_text(encoding="utf-8")

    for phrase in (
        "`./scripts/dx_smoke.sh` on Python 3.12",
        "`./scripts/verify_vercel_ai_sdk_example.sh` on Python 3.12",
        "`./scripts/ci_self_host_smoke.sh` on Python 3.12",
        "Pytest tiers:",
    ):
        assert phrase in body
    for command in CI_NON_PYTEST_CHECKS[2:]:
        assert f"`{command}`" in body


def test_ci_workflow_runs_documented_validation_surface() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    run_commands = {
        stripped.removeprefix("run: ").strip()
        for line in workflow.splitlines()
        if (stripped := line.strip()).startswith("run: ")
    }

    assert "branches: [main]" in workflow
    assert "python-version: ['3.11', '3.12']" in workflow
    assert workflow.count("if: matrix.python-version == '3.12'") == 3
    for command in ("uv sync --group dev", *CI_NON_PYTEST_CHECKS, *CI_PYTEST_TIERS):
        assert command in run_commands


def test_validation_docs_use_public_surface_language() -> None:
    root = Path(__file__).resolve().parents[1]
    text = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in ("tests/README.md", "scripts/README.md", "docs/stability.md")
    )

    assert "public-surface checks" in text
    assert "public surface changes" in text
    assert "Search pipeline internals" in text
    assert "Docs, packaging, exports" not in text
    assert "public export surface" not in text
    assert "Orchestrator internals" not in text


def test_search_pipeline_internals_coordinate_through_typed_stage_state() -> None:
    root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    for path in (
        *(root / "src" / "rag_core" / "search").rglob("*.py"),
        *(root / "tests").glob("test_pipeline*.py"),
    ):
        source = path.read_text(encoding="utf-8")
        assert "query.extra" not in source
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "extra":
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")

    assert offenders == []
    pipeline_types = (
        root / "src" / "rag_core" / "search" / "pipeline" / "types.py"
    ).read_text(encoding="utf-8")
    assert "extra: dict" not in pipeline_types
    assert "class PipelineStageState" in pipeline_types


def test_pipeline_docs_do_not_reintroduce_orchestration_language() -> None:
    root = Path(__file__).resolve().parents[1]
    text = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in (
            "tests/test_retrieval_golden_path.py",
            "tests/test_metadata_filter.py",
            "src/rag_core/documents/converters/base.py",
        )
    )

    assert "search orchestration" not in text
    assert "pin orchestration" not in text
    assert "through the orchestrator" not in text
    assert "fallback orchestration" not in text


def test_search_pipeline_runner_replaced_internal_orchestrator_name() -> None:
    root = Path(__file__).resolve().parents[1]
    text = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in (
            "docs/stability.md",
            "src/rag_core/core.py",
            "src/rag_core/_engine/core_assembly.py",
            "src/rag_core/search/pipeline_runner.py",
            "src/rag_core/search/pipeline_runner_defaults.py",
            "tests/test_search_pipeline_runner.py",
            "tests/test_default_search_pipeline.py",
            "tests/test_metadata_filter.py",
            "tests/test_pipeline_error_traces.py",
        )
    )

    assert "SearchPipelineRunner" in text
    assert "rag_core.search.pipeline_runner" in text
    assert "SearchOrchestrator" not in text
    assert "search orchestrator" not in text
    assert "Orchestration exists" not in text
    assert "rag_core.search.searcher" not in text
    assert "searcher_pipeline" not in text


def test_operational_docs_do_not_reintroduce_orchestration_language() -> None:
    root = Path(__file__).resolve().parents[1]
    text = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in ("docs/self-host.md", "scripts/README.md", "pyproject.toml")
    )

    assert "resume-after-crash recovery" in text
    assert "scripts are **coordination** only" in text
    assert "resume-after-crash orchestration" not in text
    assert "scripts are **orchestration** only" not in text
    assert "wiring-only orchestration checks" not in text
    assert "orchestrators" not in text


def test_pytest_markers_use_adapter_provider_language() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "real in-memory adapters" in pyproject
    assert "public-surface checks" in pyproject
    assert "real in-memory backends" not in pyproject
    assert "docs, packaging, exports" not in pyproject
