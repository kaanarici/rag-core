from __future__ import annotations
import argparse
from pathlib import Path
import tarfile
import zipfile

REQUIRED_SDIST_SUFFIXES = frozenset(
    {
        "README.md",
        ".env.example",
        "docs/release.md",
        "docs/api.md",
        "docs/architecture.md",
        "docs/cli.md",
        "docs/dependability.md",
        "docs/evaluation.md",
        "docs/formats.md",
        "docs/migration.md",
        "docs/tracing.md",
        "examples/README.md",
        "examples/configured_retrieval.py",
        "examples/configured_eval.py",
        "examples/minimal_app.py",
        "examples/retrieval_eval.py",
        "examples/demo_corpus/corpus_lifecycle.md",
        "scripts/dx_smoke.sh",
        "scripts/github_install_smoke.sh",
        "scripts/landing_check.sh",
        "scripts/public_checkout_smoke.sh",
        "scripts/validate_provider_fixtures.sh",
        "scripts/wheel_smoke.py",
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

REQUIRED_WHEEL_MEMBERS = {
    "rag_core/py.typed",
    "rag_core/__init__.pyi",
    "rag_core/search/__init__.pyi",
}
FORBIDDEN_FRAGMENTS = (
    "_engine/core_config_cli.py",
    "cli/core_runtime.py",
    "cli/defaults.py",
    "cli/output.py",
    "config/__init__.pyi",
    "events/__init__.pyi",
    "providers/chunk_lookup.py",
    "providers/rerank_results.py",
    "providers/reranker.py",
    "pdf_inspector",
    "converters/registry_",
    "converters/format_support",
    "manifest/preview/",
    "manifest/reconciliation/",
    "licensed_source_registry.py",
    "embedding_input_types.py",
    "provider_package_names.py",
    "documents/http_errors.py",
    "ingest/lifecycle.py",
    "ingest/progress.py",
    "ingest/payloads.py",
    "rag_core/runtime/",
    "rag_core/search/pipeline/",
    "query_plan",
    "lexical_sidecar",
    "embedding_cache",
    "chunk_context_cache",
    "cached_embedding",
    "pgvector",
    "memory_store",
    "langchain",
    "openai_agents",
    "zeroentropy",
    "providers/registry.py",
    "chunking/semantic.py",
    "chunking/code.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    args = parser.parse_args()
    wheels = list(args.dist_dir.glob("*.whl"))
    sdists = list(args.dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise SystemExit("Expected one wheel and one source archive")
    with zipfile.ZipFile(wheels[0]) as archive:
        names = set(archive.namelist())
        assert REQUIRED_WHEEL_MEMBERS <= names
        assert not [n for n in names if any(part in n for part in FORBIDDEN_FRAGMENTS)]
        root = Path(__file__).resolve().parents[1] / "src"
        expected = {
            p.relative_to(root).as_posix()
            for p in (root / "rag_core").rglob("*")
            if p.suffix in {".py", ".pyi"}
        }
        actual = {
            n
            for n in names
            if n.startswith("rag_core/") and n.endswith((".py", ".pyi"))
        }
        assert actual == expected, (actual - expected, expected - actual)
        assert all(archive.read(n) == (root / n).read_bytes() for n in actual)
    with tarfile.open(sdists[0]) as archive:
        members = {
            "/".join(member.name.split("/")[1:]): member
            for member in archive.getmembers()
        }
        names = set(members)
        assert REQUIRED_SDIST_SUFFIXES <= names
        assert not (FORBIDDEN_SDIST_SUFFIXES & names)
        for name in REQUIRED_SDIST_SUFFIXES:
            member = archive.extractfile(members[name])
            assert member is not None
            with member:
                assert member.read() == (root.parent / name).read_bytes(), name
    print(
        "Wheel source, retired-module exclusions, typing artifacts, and source archive passed"
    )


if __name__ == "__main__":
    main()
