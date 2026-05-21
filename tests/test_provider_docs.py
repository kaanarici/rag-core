from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

pytestmark = [pytest.mark.meta]


def test_provider_docs_match_current_install_story() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    provider_docs = Path("docs/providers.md").read_text(encoding="utf-8")

    assert 'uv add "rag-core @ git+https://github.com/kaanarici/rag-core.git"' in readme
    assert "QdrantConfig" in provider_docs
    assert "default wheel** ships **Qdrant**" in provider_docs


def test_readme_first_run_names_module_smokes() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "Not installed into the wheel" in readme

    required_commands = (
        "uv run python -m examples.minimal_app",
        "uv run python -m examples.search_endpoint",
        "uv run python -m examples.source_ingest",
        "uv run python -m examples.retrieval_eval",
    )
    for command in required_commands:
        assert command in readme


def test_readme_documents_checkout_examples_and_eval_contracts() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "not installed into the wheel" in readme.lower()
    assert "keep cases in your app repo" in readme
    assert 'cases = load_cases(Path("cases.jsonl"))' in readme
    assert "from rag_core.contracts import parse_search_user_documents_request" in readme
    assert "RAGCore.retrieve_context(...)" in readme


def test_readme_shows_installed_no_key_library_path() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "from rag_core.demo import build_demo_core" in readme
    assert 'build_demo_core(collection="quickstart")' in readme
    assert 'qdrant_location="./rag-core-qdrant"' in readme
    assert "works from an installed wheel without API keys" in readme
    assert "pip install rag-core" in readme or "pip install`" in readme


def test_readme_documents_all_declared_extras() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    extras = set(pyproject["project"]["optional-dependencies"])
    readme = Path("README.md").read_text(encoding="utf-8")
    provider_docs = Path("docs/providers.md").read_text(encoding="utf-8")

    assert extras, "expected at least one optional dependency group"
    for extra in extras:
        assert f"`{extra}`" in readme
        assert f"`{extra}`" in provider_docs


def test_readme_distinguishes_json_stdout_from_jsonl_files() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "Batch ingest commands" in readme
    assert "one JSON object per record" in readme
    assert "JSONL" in readme


def test_examples_reuse_packaged_demo_core() -> None:
    example_paths = (
        Path("examples/minimal_app.py"),
        Path("examples/chatbot_context.py"),
        Path("examples/search_endpoint.py"),
        Path("examples/retrieval_eval.py"),
        Path("examples/source_ingest.py"),
    )

    for path in example_paths:
        text = path.read_text(encoding="utf-8")
        assert "from rag_core.demo import build_demo_core" in text
        assert "DemoEmbeddingProvider" not in text
        assert "DemoSparseEmbedder" not in text


def test_readme_first_run_uses_a_user_owned_folder() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    quickstart_command = (
        'uv run rag-core local-search /tmp/rag-core-quickstart "How can invoices be paid?" --json'
    )

    assert quickstart_command in readme
    assert "local-search" in readme


def test_ci_runs_installed_consumer_wheel_smoke() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    smoke = Path("scripts/wheel_smoke.py").read_text(encoding="utf-8")

    assert "uv run python scripts/wheel_smoke.py" in workflow
    assert "consumer-app" in smoke
    assert "from rag_core.demo import build_demo_core" in smoke
    assert "qdrant_location=str(qdrant_location)" in smoke
    assert "rag_core imported from checkout" in smoke
    assert "_write_no_key_sources" in smoke
    assert '"expected retrieved context citations for all source types"' in smoke
    assert '"expected persistent Qdrant citations"' in smoke


def test_custom_provider_docs_name_all_extension_points() -> None:
    docs = Path("docs/providers.md").read_text(encoding="utf-8")

    required_terms = (
        "rag_core.search.types.EmbeddingProvider",
        "EMBEDDING_PROVIDERS",
        "rag_core.search.types.SparseEmbedder",
        "SPARSE_EMBEDDERS",
        "rag_core.search.types.RerankerProvider",
        "RERANKER_PROVIDERS",
        "rag_core.search.types.VectorStore",
        "VECTOR_STORES",
        "StoreCapabilities",
        "rag_core.documents.OcrProvider",
        "OCR_PROVIDERS",
        "rag_core.search.types.SearchSidecar",
        "SEARCH_SIDECARS",
        "rag_core.search.providers.EmbeddingCache",
        "EMBEDDING_CACHES",
        "rag_core.search.providers.ChunkContextCache",
        "CHUNK_CONTEXT_CACHES",
        "rag_core.documents.chunking.ChunkingStrategy",
        "CHUNKING_STRATEGIES",
        "rag_core.documents.converters.BaseConverter",
        "ConversionResult",
    )

    for term in required_terms:
        assert term in docs


def test_custom_provider_docs_distinguish_config_and_injection_surfaces() -> None:
    docs = Path("docs/providers.md").read_text(encoding="utf-8")

    required_terms = (
        "A registry entry does not automatically make a category selectable from `RAGCoreConfig`.",
        "Dense embeddings",
        "RAGCoreConfig.embedding",
        "Sparse embeddings",
        "RAGCore(sparse_embedder=...)",
        "OCR",
        "RAGCore(ocr_provider=...)`; no `RAGCoreConfig` field",
        "Event sinks",
        "injection surface",
    )
    for term in required_terms:
        assert term in docs


def test_custom_provider_docs_explain_support_level_diagnostics() -> None:
    docs = Path("docs/providers.md").read_text(encoding="utf-8")

    required_terms = (
        "Support levels and diagnostics",
        "rag-core doctor --json",
        "Dense embeddings",
        "Sparse embeddings",
        "Rerankers",
        "OCR",
        "Contextualizers",
        "Caches",
        "Search sidecars",
        "Event sinks",
        "Vector stores",
    )
    for term in required_terms:
        assert term in docs


def test_provider_output_shape_audit_stays_visible() -> None:
    docs = Path("docs/providers.md").read_text(encoding="utf-8")

    required_terms = (
        "Checked against provider docs on 2026-05-20.",
        "OpenAI embeddings",
        "Voyage embeddings",
        "ZeroEntropy embeddings",
        "Cohere rerank",
        "Voyage rerank",
        "ZeroEntropy rerank",
        "Qdrant Query API",
        "TurboPuffer query",
        "Mistral OCR",
        "Gemini command OCR",
        "adapter/parser tests unless a test is marked `live`",
    )
    for term in required_terms:
        assert term in docs
