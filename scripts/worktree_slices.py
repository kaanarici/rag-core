from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


FAST_PYTEST = (
    'uv run pytest -q -m "not live and not eval and not eval_harness '
    'and not provider_contract and not integration"'
)


@dataclass(frozen=True)
class ChangedPath:
    status: str
    path: str


@dataclass(frozen=True)
class SliceDefinition:
    key: str
    title: str
    purpose: str
    validation: tuple[str, ...]
    exact: tuple[str, ...] = ()
    prefixes: tuple[str, ...] = ()


SLICE_DEFINITIONS: tuple[SliceDefinition, ...] = (
    SliceDefinition(
        key="validation-tooling",
        title="Validation Tooling",
        purpose="Local gates, packaging checks, and maintainer scripts.",
        validation=(
            "bash -n scripts/landing_check.sh",
            "uv run pytest -q tests/test_validation_product_shape.py tests/test_packaging_manifest.py",
            "./scripts/landing_check.sh --quick",
        ),
        exact=("tests/README.md", "tests/test_validation_product_shape.py"),
        prefixes=("scripts/",),
    ),
    SliceDefinition(
        key="public-docs-examples",
        title="Public Docs And Examples",
        purpose="README, product docs, and copyable example programs.",
        validation=(
            "uv run pytest -q tests/test_product_docs_shape.py tests/test_examples.py",
            "./scripts/dx_smoke.sh",
        ),
        exact=("README.md",),
        prefixes=("docs/", "examples/"),
    ),
    SliceDefinition(
        key="repo-packaging-hygiene",
        title="Repo And Packaging Hygiene",
        purpose="Checkout metadata, local-only files, Docker, compose, and sdist shape.",
        validation=(
            "uv run pytest -q tests/test_packaging_manifest.py",
            "uv build && uv run python scripts/check_dist_artifacts.py",
        ),
        exact=("MANIFEST.in", "pyproject.toml", "Dockerfile", "compose.yaml"),
        prefixes=(".env", ".git", "dev/"),
    ),
    SliceDefinition(
        key="public-api-cli-config",
        title="Public API, CLI, And Config",
        purpose="Import aggregators, command parsing/output, config objects, and tool contracts.",
        validation=(
            "uv run pytest -q tests/test_public_contracts.py tests/test_cli.py tests/test_cli_product_shape.py",
            "./scripts/dx_smoke.sh",
        ),
        prefixes=(
            "src/rag_core/__init__",
            "src/rag_core/cli",
            "src/rag_core/config/",
            "src/rag_core/contracts/",
            "src/rag_core/demo.py",
            "src/rag_core/archive_sources.py",
            "src/rag_core/local_eval",
            "src/rag_core/local_search",
            "src/rag_core/local_sources.py",
            "src/rag_core/manifest_reconciliation.py",
            "src/rag_core/quickstart.py",
            "src/rag_core/remote_document_keys.py",
            "src/rag_core/remote_sources.py",
        ),
    ),
    SliceDefinition(
        key="core-ingest-manifest",
        title="Core Ingest And Manifest",
        purpose="RAGCore ingest, source identity, lifecycle, manifests, and reconciliation.",
        validation=(
            "uv run pytest -q tests/test_core_lifecycle.py tests/test_manifest_product_shape.py tests/test_ingest_lifecycle_product_shape.py",
            "./scripts/dx_smoke.sh",
        ),
        prefixes=(
            "src/rag_core/core",
            "src/rag_core/facade/",
            "src/rag_core/fetch",
            "src/rag_core/ingest",
            "src/rag_core/local_ingest",
            "src/rag_core/manifest",
            "src/rag_core/remote_",
        ),
    ),
    SliceDefinition(
        key="documents-parsing",
        title="Documents, Parsing, And Chunking",
        purpose="Converters, OCR, chunking strategies, document runtime, and parser fixtures.",
        validation=(
            "uv run pytest -q tests/test_converter_product_shape.py tests/test_document_runtime_product_shape.py tests/test_real_document_fixtures.py",
            "./scripts/dx_smoke.sh",
        ),
        prefixes=("src/rag_core/documents/",),
    ),
    SliceDefinition(
        key="providers-integrations",
        title="Providers And Integrations",
        purpose="Embedding, vector-store, reranker, cache, sidecar, LangChain, and Agents seams.",
        validation=(
            "uv run pytest -q tests/test_provider_registry.py tests/test_vector_store_contract.py tests/test_langchain_integration.py tests/test_openai_agents_tool.py",
            "uv run python scripts/wheel_smoke.py",
        ),
        prefixes=(
            "src/rag_core/integrations/",
            "src/rag_core/provider_",
            "src/rag_core/search/providers/",
        ),
    ),
    SliceDefinition(
        key="search-retrieval-pipeline",
        title="Search And Retrieval Pipeline",
        purpose="Query plans, search pipeline stages, context packing, lexical sidecar, and scoring.",
        validation=(
            "uv run pytest -q tests/test_search_pipeline_runner.py tests/test_query_plan.py tests/test_retrieval_golden_path.py",
            "./scripts/dx_smoke.sh",
        ),
        prefixes=(
            "src/rag_core/retrieval_",
            "src/rag_core/search/",
        ),
    ),
    SliceDefinition(
        key="runtime-self-host",
        title="Runtime And Self Host",
        purpose="Optional HTTP runtime that stays thin over RAGCore.",
        validation=(
            "uv run pytest -q tests/test_runtime_http.py tests/test_runtime_product_shape.py",
            "./scripts/ci_self_host_smoke.sh",
        ),
        prefixes=("src/rag_core/runtime", "src/rag_core/runtime_", "docs/self-host"),
    ),
    SliceDefinition(
        key="events-traces-evals",
        title="Events, Traces, And Evals",
        purpose="Event payloads, trace summaries, eval cases, and reporting.",
        validation=(
            "uv run pytest -q tests/test_events.py tests/test_trace_product_shape.py tests/test_evals_runner.py",
            "uv run pytest -q tests/evals/test_retrieval_eval_pr.py",
        ),
        prefixes=("src/rag_core/events/", "src/rag_core/evals/", "tests/evals/"),
    ),
    SliceDefinition(
        key="tests-fixtures-support",
        title="Tests, Fixtures, And Support",
        purpose="Test harness, fakes, fixtures, and product-shape coverage not owned above.",
        validation=("uv run pytest -q", FAST_PYTEST),
        prefixes=("tests/",),
    ),
    SliceDefinition(
        key="uncategorized",
        title="Uncategorized",
        purpose="Paths that need human routing before landing.",
        validation=("uv run ruff check .", "uv run mypy src tests examples"),
    ),
)

SLICE_BY_KEY = {definition.key: definition for definition in SLICE_DEFINITIONS}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify the local git worktree into reviewable rag-core landing slices."
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        help="classify only staged changes using git diff --cached --name-status",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON instead of Markdown",
    )
    parser.add_argument(
        "--slice",
        dest="slice_key",
        help="show one slice by key, for example validation-tooling",
    )
    parser.add_argument(
        "--list-slices",
        action="store_true",
        help="print known slice keys and exit",
    )
    parser.add_argument(
        "--fail-on-uncategorized",
        action="store_true",
        help="exit nonzero when any changed path lands in the Uncategorized slice",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=18,
        help="maximum files to list per slice in Markdown output",
    )
    args = parser.parse_args()

    if args.list_slices:
        print("\n".join(definition.key for definition in SLICE_DEFINITIONS))
        return
    if args.slice_key is not None:
        validate_slice_key(args.slice_key)

    changes = read_changed_paths(staged=args.staged)
    grouped = group_changes(changes)
    if args.fail_on_uncategorized:
        ensure_no_uncategorized(grouped)
    if args.slice_key is not None:
        grouped = select_slice(grouped, args.slice_key)
    mode = "staged" if args.staged else "worktree"
    if args.json:
        print(json.dumps(serialize_groups(grouped, mode=mode), indent=2))
    else:
        print(render_markdown(grouped, mode=mode, max_files=args.max_files))


def read_changed_paths(*, staged: bool) -> list[ChangedPath]:
    if staged:
        completed = subprocess.run(
            ["git", "diff", "--cached", "--name-status"],
            check=True,
            capture_output=True,
            text=True,
        )
        return parse_name_status(completed.stdout.splitlines())

    completed = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_porcelain(completed.stdout.splitlines())


def parse_porcelain(lines: Iterable[str]) -> list[ChangedPath]:
    changes: list[ChangedPath] = []
    for line in lines:
        if not line:
            continue
        status = line[:2].strip() or "?"
        path = _rename_target(line[3:])
        changes.append(ChangedPath(status=status[0], path=path))
    return changes


def parse_name_status(lines: Iterable[str]) -> list[ChangedPath]:
    changes: list[ChangedPath] = []
    for line in lines:
        if not line:
            continue
        parts = line.split("\t")
        status = parts[0][0]
        path = parts[-1]
        changes.append(ChangedPath(status=status, path=path))
    return changes


def group_changes(changes: Iterable[ChangedPath]) -> dict[str, list[ChangedPath]]:
    grouped: dict[str, list[ChangedPath]] = {definition.key: [] for definition in SLICE_DEFINITIONS}
    for change in sorted(changes, key=lambda item: item.path):
        grouped[classify_path(change.path)].append(change)
    return grouped


def validate_slice_key(slice_key: str) -> None:
    if slice_key in SLICE_BY_KEY:
        return
    known = ", ".join(definition.key for definition in SLICE_DEFINITIONS)
    raise SystemExit(f"unknown slice {slice_key!r}; known slices: {known}")


def select_slice(
    grouped: dict[str, list[ChangedPath]], slice_key: str
) -> dict[str, list[ChangedPath]]:
    validate_slice_key(slice_key)
    return {definition.key: grouped.get(definition.key, []) for definition in SLICE_DEFINITIONS if definition.key == slice_key}


def ensure_no_uncategorized(grouped: dict[str, list[ChangedPath]]) -> None:
    uncategorized = grouped.get("uncategorized", [])
    if not uncategorized:
        return
    paths = "\n".join(f"- {change.path}" for change in uncategorized)
    raise SystemExit(f"uncategorized changed paths need routing:\n{paths}")


def classify_path(path: str) -> str:
    test_slice = _classify_test_path(path)
    if test_slice is not None:
        return test_slice
    if path.startswith("docs/templates/"):
        return "repo-packaging-hygiene"
    if path.startswith("docs/self-host"):
        return "runtime-self-host"
    for definition in SLICE_DEFINITIONS:
        if definition.key in {"tests-fixtures-support", "uncategorized"}:
            continue
        if path in definition.exact or path.startswith(definition.prefixes):
            return definition.key
    return "uncategorized"


def render_markdown(
    grouped: dict[str, list[ChangedPath]], *, mode: str, max_files: int
) -> str:
    total = sum(len(changes) for changes in grouped.values())
    lines = [
        "# rag-core worktree slices",
        "",
        f"Mode: {mode}",
        f"Changed paths: {total}",
        "",
    ]
    if total == 0:
        lines.append("No changed paths.")
        return "\n".join(lines)

    lines.append("Recommended review order follows the sections below.")
    for definition in SLICE_DEFINITIONS:
        changes = grouped.get(definition.key, [])
        if not changes:
            continue
        statuses = _status_summary(changes)
        lines.extend(
            [
                "",
                f"## {definition.title} ({len(changes)} paths; {statuses})",
                "",
                definition.purpose,
                "",
                "Validation:",
            ]
        )
        lines.extend(f"- `{command}`" for command in definition.validation)
        lines.extend(["", "Files:"])
        listed = changes if max_files <= 0 else changes[:max_files]
        lines.extend(f"- `{change.status}` {change.path}" for change in listed)
        if max_files > 0 and len(changes) > max_files:
            lines.append(f"- ... {len(changes) - max_files} more")

    return "\n".join(lines)


def serialize_groups(grouped: dict[str, list[ChangedPath]], *, mode: str) -> dict[str, object]:
    slices: list[dict[str, object]] = []
    changed_paths = 0
    for definition in SLICE_DEFINITIONS:
        changes = grouped.get(definition.key, [])
        if not changes:
            continue
        changed_paths += len(changes)
        slices.append(
            {
                "key": definition.key,
                "title": definition.title,
                "count": len(changes),
                "statuses": dict(Counter(change.status for change in changes)),
                "validation": list(definition.validation),
                "files": [
                    {"status": change.status, "path": change.path} for change in changes
                ],
            }
        )
    return {
        "mode": mode,
        "changed_paths": changed_paths,
        "slices": slices,
    }


def _classify_test_path(path: str) -> str | None:
    if path == "tests/README.md":
        return "validation-tooling"
    if path.startswith(("tests/fixtures/", "tests/support/", "tests/conftest.py")):
        return "tests-fixtures-support"
    if not path.startswith("tests/"):
        return None

    name = Path(path).name
    if any(token in name for token in ("validation", "packaging", "product_docs")):
        return "validation-tooling"
    if any(token in name for token in ("cli", "public_contract", "tool_contract")):
        return "public-api-cli-config"
    if any(token in name for token in ("runtime", "self_host")):
        return "runtime-self-host"
    if any(token in name for token in ("event", "trace", "eval")):
        return "events-traces-evals"
    if any(
        token in name
        for token in (
            "document",
            "converter",
            "chunk",
            "contextualizer",
            "format",
            "ocr",
            "parser",
            "pdf",
            "office",
        )
    ):
        return "documents-parsing"
    if any(
        token in name
        for token in (
            "cache",
            "embedding",
            "integration",
            "langchain",
            "openai_agents",
            "provider",
            "qdrant",
            "rerank",
            "sparse",
            "store",
            "turbopuffer",
            "vector_store",
            "zeroentropy",
        )
    ):
        return "providers-integrations"
    if any(
        token in name
        for token in (
            "context_pack",
            "indexer",
            "metadata_filter",
            "pipeline",
            "query_plan",
            "retrieval",
            "search",
            "sidecar",
        )
    ):
        return "search-retrieval-pipeline"
    if any(token in name for token in ("core", "ingest", "manifest", "remote")):
        return "core-ingest-manifest"
    return "tests-fixtures-support"


def _rename_target(path: str) -> str:
    if " -> " in path:
        return path.rsplit(" -> ", maxsplit=1)[1]
    return path


def _status_summary(changes: Iterable[ChangedPath]) -> str:
    counts = Counter(change.status for change in changes)
    return ", ".join(f"{status}:{count}" for status, count in sorted(counts.items()))


if __name__ == "__main__":
    main()
