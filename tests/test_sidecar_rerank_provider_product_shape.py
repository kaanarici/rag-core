from __future__ import annotations

import ast
from pathlib import Path

from rag_core.retrieval_defaults import (
    DEFAULT_SEARCH_LIMIT,
)
from rag_core.search.lexical_sidecar import (
    PORTABLE_LEXICAL_SIDECAR_PROVIDER,
    SEARCH_SIDECAR_PROVIDER_ORDER,
)

CANONICAL_LAUNCH_GATES = (
    "uv run ruff check .",
    "uv run mypy src tests examples",
    "uv run pytest -q",
    "./scripts/dx_smoke.sh",
    "./scripts/ci_self_host_smoke.sh",
    "uv build",
    "uv run python scripts/check_dist_artifacts.py",
    "uv run python scripts/wheel_smoke.py",
)


def test_search_sidecar_provider_defaults_have_single_sidecar_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/search/lexical_sidecar.py",
            "src/rag_core/search/providers/provider_category_diagnostics.py",
            "src/rag_core/cli_doctor_output.py",
        )
    }

    assert PORTABLE_LEXICAL_SIDECAR_PROVIDER == "portable_lexical"
    assert SEARCH_SIDECAR_PROVIDER_ORDER == ("portable_lexical",)
    lexical_sidecar = sources["src/rag_core/search/lexical_sidecar.py"]
    assert (
        "PORTABLE_LEXICAL_SIDECAR_PROVIDER = PortableLexicalSidecar.provider_name"
        in (lexical_sidecar)
    )
    diagnostics = sources[
        "src/rag_core/search/providers/provider_category_diagnostics.py"
    ]
    doctor_output = sources["src/rag_core/cli_doctor_output.py"]
    assert "SEARCH_SIDECAR_PROVIDER_ORDER" in diagnostics
    assert "SEARCH_SIDECAR_PROVIDER_ORDER" in doctor_output
    assert "_SEARCH_SIDECAR_PROVIDER_ALIASES" not in diagnostics
    assert 'known=("portable_lexical",)' not in diagnostics
    assert '("search_sidecar", ("portable_lexical",))' not in doctor_output





def test_test_reranker_helpers_use_named_search_limit_default() -> None:
    root = Path(__file__).resolve().parents[1]
    files = (
        "tests/support/fakes.py",
        "tests/test_reranker_registry_contract.py",
        "tests/test_pipeline.py",
        "tests/test_runtime_http.py",
        "tests/test_rerank_diagnostics.py",
    )
    offenders: list[str] = []

    for filename in files:
        path = root / filename
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = node.args.args
            defaults = node.args.defaults
            if not defaults:
                continue
            default_start = len(args) - len(defaults)
            for arg, default in zip(args[default_start:], defaults):
                if (
                    arg.arg == "top_k"
                    and isinstance(default, ast.Constant)
                    and default.value == DEFAULT_SEARCH_LIMIT
                ):
                    offenders.append(f"{filename}:{node.lineno}")

    assert offenders == []
