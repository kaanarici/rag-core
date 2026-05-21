from __future__ import annotations

import tomllib
from pathlib import Path


def test_provider_docs_match_current_install_story() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    vector_store_docs = Path("docs/providers/vector-stores.md").read_text(encoding="utf-8")

    assert 'uv add "rag-core @ git+https://github.com/kaanarici/rag-core.git"' in readme
    assert "git+https://github.com/kaanarici/rag-core.git" in vector_store_docs


def test_readme_first_run_names_module_smokes() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "The `examples/` modules below are checkout examples." in readme
    assert "They are not installed into the wheel." in readme

    required_commands = (
        "uv run python -m examples.minimal_app",
        "uv run python -m examples.search_endpoint",
        "uv run python -m examples.source_ingest",
        "uv run python -m examples.retrieval_eval",
    )
    for command in required_commands:
        assert command in readme


def test_docs_mark_checkout_examples_and_installed_import_surfaces() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    eval_docs = Path("docs/evals/retrieval-quality.md").read_text(encoding="utf-8")
    vercel_docs = Path("docs/integrations/vercel-ai-sdk-tools.md").read_text(
        encoding="utf-8"
    )

    for docs in (readme, eval_docs, vercel_docs):
        assert "not installed into the wheel" in docs

    assert "Installed-package users should keep eval cases in their own app repo" in eval_docs
    assert "cases = load_cases(Path(\"cases.jsonl\"))" in eval_docs
    assert "from rag_core.contracts import parse_search_user_documents_request" in vercel_docs
    assert "RAGCore.retrieve_context(...)" in vercel_docs


def test_readme_shows_installed_no_key_library_path() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "from rag_core.demo import build_demo_core" in readme
    assert 'build_demo_core(collection="quickstart")' in readme
    assert 'qdrant_location="./rag-core-qdrant"' in readme
    assert "works from an installed wheel without API keys" in readme


def test_readme_documents_all_declared_extras() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    extras = set(pyproject["project"]["optional-dependencies"])
    readme = Path("README.md").read_text(encoding="utf-8")
    provider_docs = Path("docs/providers/custom-providers.md").read_text(encoding="utf-8")

    assert extras == {
        "semantic",
        "html",
        "rerank",
        "voyage",
        "zeroentropy",
        "turbopuffer",
        "opentelemetry",
        "anthropic",
        "langchain",
        "openai-agents",
    }
    for extra in extras:
        assert f"`{extra}`" in readme or f"--extra {extra}" in readme
        assert f"`{extra}`" in provider_docs


def test_readme_distinguishes_json_stdout_from_jsonl_files() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "Batch ingest commands" in readme
    assert "one JSON object per record" in readme
    assert "`--events-jsonl`, manifest files, eval case files, and batch ingest stdout are JSONL" in readme


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


def test_readme_first_run_uses_a_user_owned_folder_before_demo_corpus() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    examples = Path("examples")
    corpus = examples / "demo_corpus"
    quickstart_command = (
        'uv run rag-core local-search /tmp/rag-core-quickstart "How can invoices be paid?" --json'
    )
    demo_trace_command = (
        'uv run rag-core local-search examples/demo_corpus "corpus lifecycle" \\'
    )

    assert quickstart_command in readme
    assert readme.index(quickstart_command) < readme.index(demo_trace_command)
    assert "It indexes up to 200 supported files by default" not in readme
    assert "It indexes a folder into embedded Qdrant" not in readme
    assert "local-search` indexes a folder into embedded Qdrant" in readme
    assert "`--max-files`" in readme
    assert corpus.is_dir()
    assert sorted(path.suffix for path in corpus.iterdir()) == [".md", ".md", ".md"]


def test_ci_smokes_turbopuffer_extra_wheel() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    docs = Path("docs/providers/vector-stores.md").read_text(encoding="utf-8")

    assert "[turbopuffer]" in workflow
    assert "--vector-store turbopuffer" in workflow
    assert "TurboPuffer extra install" in docs


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
    docs = Path("docs/providers/custom-providers.md").read_text(encoding="utf-8")

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
    docs = Path("docs/providers/custom-providers.md").read_text(encoding="utf-8")

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
    docs = Path("docs/providers/custom-providers.md").read_text(encoding="utf-8")

    required_terms = (
        "Support Levels And Diagnostics",
        "rag-core doctor --json",
        "Dense embeddings",
        "Sparse embeddings",
        "Rerankers",
        "OCR",
        "Contextualizers",
        "Caches",
        "Search sidecars",
        "Event sinks",
        "Vector-store diagnostics",
    )
    for term in required_terms:
        assert term in docs


def test_provider_output_shape_audit_stays_visible() -> None:
    docs = Path("docs/providers/provider-output-shapes.md").read_text(encoding="utf-8")

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
        "live conformance scripts",
    )
    for term in required_terms:
        assert term in docs
