"""Context-pack ownership, projection, and layering invariants.

The context pack exposes two projections -- app-facing (``to_payload`` /
``as_text``) and prompt-safe (``to_prompt_payload`` / ``as_prompt_text``) -- and a
single response-payload builder, while ordering (``context_order``) stays a
presentation/runtime concern and never leaks into the facade, engine, or
integration protocol. Structural ownership is asserted via symbol resolution and
the import graph so it survives module merges/renames; the product-voice
docstring guardrails (which read internal source prose) and the docs / CLI-help /
wheel-smoke honesty guardrails are preserved verbatim.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from rag_core.search.context_pack import context_pack_response_payload
from rag_core.search.context_pack_models import Context, ContextSnippet
from tests.support.source_graph import defining_modules, modules_importing, symbol_module


def _module_uses_identifier(dotted_module: str, identifier: str) -> bool:
    source = inspect.getsource(__import__(dotted_module, fromlist=["_"]))
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name) and node.id == identifier:
            return True
        if isinstance(node, ast.Attribute) and node.attr == identifier:
            return True
        if isinstance(node, ast.arg) and node.arg == identifier:
            return True
        if isinstance(node, ast.keyword) and node.arg == identifier:
            return True
    return False


def test_context_pack_schema_owns_payload_and_text_projection() -> None:
    root = Path(__file__).resolve().parents[1]
    search_root = root / "src" / "rag_core" / "search"

    # The rendering/payload helpers were merged into the schema modules; the
    # standalone files (and the integration text shim) must stay gone. Asserting
    # symbol ownership rather than file paths still fails if they reappear.
    assert symbol_module(context_pack_response_payload) == "rag_core.search.context_pack"
    assert defining_modules(
        "src/rag_core/search", name="context_pack_response_payload"
    ) == {"rag_core.search.context_pack"}
    assert (
        defining_modules("src/rag_core/search", name="_context_pack_response_payload")
        == set()
    )
    assert (
        defining_modules(
            "src/rag_core/search", name="_SupportsContextPackResponsePayload"
        )
        == set()
    )
    assert not (
        root / "src" / "rag_core" / "integrations" / "integration_context_text.py"
    ).exists()

    # Both projections are methods on the schema classes (single owner module).
    for projection in ("as_text", "as_prompt_text", "to_payload", "to_prompt_payload"):
        assert hasattr(Context, projection)
        assert hasattr(ContextSnippet, projection)
    assert symbol_module(Context) == "rag_core.search.context_pack_models"
    assert symbol_module(ContextSnippet) == "rag_core.search.context_pack_models"

    # CLI and runtime build the response through the public builder, not by
    # re-inlining the payload dict; asserting the import survives reformatting.
    builder_importers = modules_importing(
        "src/rag_core/cli/commands",
        "src/rag_core/runtime",
        predicate=lambda module: module.rsplit(".", 1)[-1]
        == "context_pack_response_payload",
    )
    assert {"rag_core.cli.commands.search", "rag_core.runtime.app"} <= set(
        builder_importers
    )

    # context_order is a presentation/runtime ordering concern: the runtime app
    # passes it, but it must not leak into the facade, engine, or integration
    # protocol layers. (AST-identifier scan: tolerant of comments/reformatting.)
    assert _module_uses_identifier("rag_core.runtime.app", "context_order")
    for layer in (
        "rag_core.facade.retrieval",
        "rag_core._engine.core_retrieval",
        "rag_core.integrations.protocols",
    ):
        assert not _module_uses_identifier(layer, "context_order")

    # Product-voice docstring guardrails (preserved): the schema uses app/prompt
    # projection language, never "model-ready"/"LLM prompts" framing.
    builder = (search_root / "context_pack.py").read_text(encoding="utf-8")
    models = (search_root / "context_pack_models.py").read_text(encoding="utf-8")
    assert "deterministic Context" in builder
    assert "model context pack" not in builder
    assert "app-facing and prompt views" in models
    assert "retrieved context snippet with app and prompt projections" in models
    assert "model-ready context block" not in models
    assert "for prompt-safe text" in models
    assert "for LLM prompts" not in models
    assert "API and prompt views" not in models
    assert "API and trace consumers" not in models
    assert "stable source ids for traces and UI/debug views" in models
    assert "rank-local citation ids for model input" in models


def test_wheel_smoke_uses_prompt_safe_context_text() -> None:
    source = Path("scripts/wheel_smoke.py").read_text(encoding="utf-8")

    assert "context.as_prompt_text()" in source
    assert "context.as_text()" not in source


def test_cli_context_help_names_prompt_safe_text_boundary() -> None:
    source = Path("src/rag_core/cli/parsers/search.py").read_text(encoding="utf-8")

    assert "context-pack JSON with prompt-safe context_text" in source
    assert "model-ready context pack JSON payload" not in source


def test_stability_docs_distinguish_context_pack_app_and_prompt_projections() -> None:
    source = Path("docs-site/content/docs/stability.mdx").read_text(encoding="utf-8")
    normalized = " ".join(source.split())

    assert "`to_payload()` / `as_text()` are app-facing" in normalized
    assert "`to_prompt_payload()` / `as_prompt_text()` are prompt-safe" in normalized
    assert "rank-local citations for model and tool responses" in normalized
