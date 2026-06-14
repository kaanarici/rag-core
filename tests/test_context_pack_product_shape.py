from pathlib import Path


def test_context_pack_schema_owns_payload_and_text_projection() -> None:
    root = Path(__file__).resolve().parents[1]
    search_root = root / "src" / "rag_core" / "search"

    assert not (search_root / "context_pack_rendering.py").exists()
    assert not (search_root / "context_pack_payload.py").exists()

    builder = (search_root / "context_pack.py").read_text(encoding="utf-8")
    models = (search_root / "context_pack_models.py").read_text(encoding="utf-8")
    cli_search = (root / "src" / "rag_core" / "cli_search.py").read_text(
        encoding="utf-8"
    )
    runtime_app = (root / "src" / "rag_core" / "runtime" / "app.py").read_text(
        encoding="utf-8"
    )
    facade = (root / "src" / "rag_core" / "facade" / "retrieval.py").read_text(
        encoding="utf-8"
    )
    core_retrieval = (
        root / "src" / "rag_core" / "_engine" / "core_retrieval.py"
    ).read_text(encoding="utf-8")
    protocols = (
        root / "src" / "rag_core" / "integrations" / "protocols.py"
    ).read_text(encoding="utf-8")
    integration_text_path = (
        root / "src" / "rag_core" / "integrations" / "integration_context_text.py"
    )
    assert "deterministic ContextPack" in builder
    assert "model context pack" not in builder
    assert "app-facing and prompt views" in models
    assert "retrieved context snippet with app and prompt projections" in models
    assert "model-ready context block" not in models
    assert "for prompt-safe text" in models
    assert "for LLM prompts" not in models
    assert "context_pack_response_payload" in builder
    assert "_context_pack_response_payload" not in builder
    assert "class _SupportsContextPackResponsePayload" not in builder
    assert "pack: ContextPack" in builder
    assert "context_pack_response_payload(pack, context_order=context_order)" in cli_search
    assert "context_pack_response_payload(" in runtime_app
    assert "context_order=retrieval_request.context_order" in runtime_app
    assert "context_order" not in facade
    assert "context_order" not in core_retrieval
    assert "context_order" not in protocols
    assert '{**pack.to_payload(), "context_text": pack.as_prompt_text()}' not in (
        cli_search + runtime_app
    )
    assert not integration_text_path.exists()
    assert "API and prompt views" not in models
    assert "API and trace consumers" not in models
    assert "stable source ids for traces and UI/debug views" in models
    assert "rank-local citation ids for model input" in models
    for method in (
        "def as_text(",
        "def as_prompt_text(",
        "def to_payload(",
        "def to_prompt_payload(",
    ):
        assert method in models


def test_wheel_smoke_uses_prompt_safe_context_text() -> None:
    source = Path("scripts/wheel_smoke.py").read_text(encoding="utf-8")

    assert "context.as_prompt_text()" in source
    assert "context.as_text()" not in source


def test_cli_context_help_names_prompt_safe_text_boundary() -> None:
    source = Path("src/rag_core/cli_search_parser.py").read_text(encoding="utf-8")

    assert "context-pack JSON with prompt-safe context_text" in source
    assert "model-ready context pack JSON payload" not in source


def test_stability_docs_distinguish_context_pack_app_and_prompt_projections() -> None:
    source = Path("docs-site/content/docs/stability.mdx").read_text(encoding="utf-8")
    normalized = " ".join(source.split())

    assert "`to_payload()` / `as_text()` are app-facing" in normalized
    assert "`to_prompt_payload()` / `as_prompt_text()` are prompt-safe" in normalized
    assert "rank-local citations for model and tool responses" in normalized
