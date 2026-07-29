from __future__ import annotations

import subprocess
from pathlib import Path

import rag_core
from rag_core.cli.output import search_hit_payload
from rag_core.contracts import SEARCH_USER_DOCUMENTS_DEFAULT_LIMIT
from rag_core.retrieval_defaults import (
    DEFAULT_CONTEXT_LIMIT,
    DEFAULT_LOCAL_SEARCH_LIMIT,
    DEFAULT_SEARCH_LIMIT,
)
from tests.support import make_search_result


def test_demo_factory_is_not_root_public_surface() -> None:
    assert "build_demo_core" not in rag_core.__all__
    assert not hasattr(rag_core, "build_demo_core")


def test_rag_core_facade_modules_live_under_facade_package() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "rag_core"

    assert not list(root.glob("core_facade_*.py"))
    assert sorted(path.name for path in (root / "facade").glob("*.py")) == [
        "__init__.py",
        "ingest.py",
        "ingest_batches.py",
        "ingest_sources.py",
        "manifest.py",
        "prepare.py",
        "retrieval.py",
    ]


def test_tracked_tree_excludes_local_plans_and_agent_material() -> None:
    root = Path(__file__).resolve().parents[1]
    tracked = tuple(
        candidate
        for path in subprocess.run(
            ["git", "ls-files"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        if (candidate := Path(path)) and (root / candidate).exists()
    )
    internal_doc_names = {"AGENTS.md", "CLAUDE.md", "CONTEXT.md", "MISSION.md"}

    assert not [path for path in tracked if path.parts and path.parts[0] == "dev"]
    assert not [
        path
        for path in tracked
        if {"plans", "research"} & {part.lower() for part in path.parts}
    ]
    assert not [path for path in tracked if path.name in internal_doc_names]
    assert not [
        path
        for path in tracked
        if path.name.lower() == "roadmap.md"
        or (
            path.suffix.lower() in {".md", ".mdx"}
            and "plan"
            in path.stem.lower().replace("-", "_").replace(" ", "_").split("_")
        )
    ]
    assert "/dev/" in (root / ".gitignore").read_text(encoding="utf-8").splitlines()


def test_self_host_configured_openai_examples_infer_known_model_dimensions() -> None:
    root = Path(__file__).resolve().parents[1]
    self_host = (root / "docs-site" / "content" / "docs" / "self-host.mdx").read_text(
        encoding="utf-8"
    )
    serve_parser = (
        root / "src" / "rag_core" / "cli" / "parsers" / "serve.py"
    ).read_text(encoding="utf-8")
    expected = (
        "--embedding-provider openai --embedding-model text-embedding-3-small"
    )
    normalized_self_host = " ".join(self_host.replace("\\\n", " ").split())

    assert expected in normalized_self_host
    assert expected in serve_parser
    assert "--embedding-model text-embedding-3-small --embedding-dimensions" not in (
        normalized_self_host + serve_parser
    )
    for line in serve_parser.splitlines():
        if "--qdrant-url http://127.0.0.1:6333" in line:
            assert "--embedding-provider openai" in line
            assert "--embedding-model text-embedding-3-small" in line
    assert (
        "`--embedding-dimensions` | Required for `demo`; use for custom or unknown models"
        in self_host
    )


def test_surface_specific_default_limits_remain_contract_values() -> None:
    assert DEFAULT_SEARCH_LIMIT == 10
    assert DEFAULT_LOCAL_SEARCH_LIMIT == 5
    assert DEFAULT_CONTEXT_LIMIT == 8
    assert SEARCH_USER_DOCUMENTS_DEFAULT_LIMIT == 5


def test_product_shape_tests_are_meta_marker_convention() -> None:
    root = Path(__file__).resolve().parents[1]
    conftest = (root / "tests" / "conftest.py").read_text(encoding="utf-8")
    product_shape_tests = sorted((root / "tests").glob("*product_shape.py"))

    assert product_shape_tests
    assert 'name.endswith("_product_shape.py")' in conftest
    assert "test_product_docs_shape.py" in conftest


def test_quickstart_install_story_matches_wheel_smoke_contract() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    wheel_smoke = Path("scripts/wheel_smoke.py").read_text(encoding="utf-8")

    assert "rag-core is not published to PyPI" in readme
    assert "git clone https://github.com/kaanarici/rag-core.git" in readme
    assert "uv sync" in readme
    assert (
        'uv pip install "rag-core @ git+https://github.com/kaanarici/rag-core.git@main"'
        in readme
    )
    assert "pip install rag-core" not in readme
    assert "uv add rag-core" not in readme
    assert "from rag_core import index" in readme
    assert 'with index("./docs") as docs:' in readme
    assert 'docs.context("How can invoices be paid?")' in readme
    assert "Document, Engine" in readme
    assert "Scope" in readme
    assert "await engine.ingest(" in readme
    assert "await engine.search(" in readme
    assert "format_evidence(result.evidence)" in readme
    assert "default mode is dense retrieval" in readme
    assert "examples.ask_folder" in readme
    assert "not semantic retrieval" in readme
    assert "--demo" in readme
    assert '[str(python), "-m", "rag_core.quickstart"]' in wheel_smoke
    assert "def _installed_cli_smoke(" in wheel_smoke
    assert "def _installed_runtime_smoke(" in wheel_smoke
    assert 'extras=("runtime",)' in wheel_smoke
    assert "installed runtime extra smoke passed" in wheel_smoke
    assert '"search"' in wheel_smoke
    assert '"eval"' in wheel_smoke
    assert "installed_cli_local_eval_cases" in wheel_smoke


def test_agent_integration_doc_shape_is_linked_from_public_docs() -> None:
    agent_doc = Path("docs-site/content/docs/agent-integration.mdx").read_text(
        encoding="utf-8"
    )
    readme = Path("README.md").read_text(encoding="utf-8")

    for shape in (
        "## Embedded Python",
        "## FastAPI endpoint",
        "## LangChain retriever and tool",
        "## OpenAI Agents tool",
        "## MCP server",
        "## Eval harness",
        "## Optional serve sidecar",
    ):
        assert shape in agent_doc
    assert "Agent integration" in readme
    assert "parse_search_user_documents_request" in agent_doc
    assert "build_langchain_retriever" in agent_doc
    assert "create_langchain_context_tool" in agent_doc
    assert "function_tool" in agent_doc
    assert "build_retrieve_context_tool" not in agent_doc
    assert "build_mcp_server" not in agent_doc
    assert "rag-core mcp" in agent_doc
    assert '"mcpServers"' in agent_doc
    assert "/v1/search" in agent_doc
    assert "/v1/search/context" in agent_doc
    assert "examples/configured_eval.py" in agent_doc
    assert "https://kaanarici.github.io/rag-core/docs/agent-integration" in readme
    assert "https://kaanarici.github.io/rag-core/docs/eval-quality" in readme
    assert "/docs/eval-quality" in agent_doc


def test_readme_keeps_prompt_formatting_optional() -> None:
    source = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in (
            "README.md",
            "docs-site/content/docs/quickstart.mdx",
            "examples/chatbot_context.py",
            "examples/minimal_app.py",
            "scripts/dx_smoke.sh",
            "src/rag_core/quickstart.py",
            "src/rag_core/search/context_pack_models.py",
            "docs-site/content/docs/stability.mdx",
        )
    )

    assert "Prompt formatting is an optional helper." in " ".join(source.split())
    assert "format_evidence(result.evidence)" in source
    assert "return model-ready context with" not in source
    assert "model-ready context" not in source
    assert "model-facing context" not in source
    assert "model-facing privacy" not in source
    assert "model payload" not in source
    assert "model prompt" not in source
    assert "model text," not in source
    assert "app/model text" not in source
    assert "for LLM calls" not in source
    assert "for LLM prompts" not in source


def test_no_key_local_eval_path_is_documented_as_folder_eval() -> None:
    source = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in (
            "README.md",
            "docs-site/content/docs/cli.mdx",
            "docs-site/content/docs/eval-quality.mdx",
            "docs-site/content/docs/quickstart.mdx",
            "scripts/dx_smoke.sh",
            "src/rag_core/cli/help_examples.py",
        )
    )

    # The no-key folder eval path is documented with its runnable command, and
    # eval cases are scored against indexed local document keys.
    assert "rag-core eval examples/demo_corpus examples/eval_cases.jsonl" in source
    assert "`document_key`" in source
    assert "expected_ids" in source
    assert "step 7: eval" in Path("scripts/dx_smoke.sh").read_text(encoding="utf-8")


def test_public_docs_do_not_claim_managed_rag_drop_in_compatibility() -> None:
    docs_dir = Path("docs-site/content/docs")
    source = "\n".join(
        text
        for text in (
            Path("README.md").read_text(encoding="utf-8"),
            *(path.read_text(encoding="utf-8") for path in docs_dir.glob("*.mdx")),
        )
    )

    assert "Ragie-compatible" not in source
    assert "rag-core parity target" not in source
    assert "drop-in Ragie compatibility" not in source


def test_release_readiness_points_to_public_checkout_smoke() -> None:
    source = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in (
            "README.md",
            "docs/release.md",
            "scripts/README.md",
            "tests/README.md",
        )
    )

    assert "./scripts/public_checkout_smoke.sh --package" in source
    assert "./scripts/github_install_smoke.sh https://github.com/kaanarici/rag-core.git main" in source
    assert "without local-only files" in source


def test_published_container_commands_opt_into_non_loopback_bind() -> None:
    root = Path(__file__).resolve().parents[1]
    compose = (root / "compose.yaml").read_text(encoding="utf-8")
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")

    if "0.0.0.0" in compose:
        assert "--bind-non-loopback" in compose
    if "0.0.0.0" in dockerfile:
        assert "--bind-non-loopback" in dockerfile


def test_search_hit_payload_matches_ragie_scored_chunk_fields() -> None:
    hit = make_search_result(
        document_id="doc-1",
        document_key="docs/guide.md",
        score=0.88,
    )
    payload = search_hit_payload(hit)
    assert payload["id"] == hit.id
    assert payload["text"] == hit.text
    assert payload["score"] == 0.88
    assert payload["document_id"] == "doc-1"
    assert payload["document_key"] == "docs/guide.md"
    assert isinstance(payload["metadata"], dict)
