"""Single-owner contracts for OCR and contextualizer provider names.

Provider-name and default-model constants are a deliberate single-owner
convention so the CLI doctor, diagnostics, and runtime all agree on one set of
names. These constants are plain module-level strings (no ``__module__``), so
ownership is proved by value equality plus an AST single-definition scan and an
import-graph check that the cross-layer consumers reach the names through the
owner. (Previously this scraped hand-pinned source files for literal provider
strings, which froze the layout.)
"""

from __future__ import annotations

import ast

from rag_core.documents.contextualizer_provider_names import (
    ANTHROPIC_CONTEXTUALIZER_ID,
    CONTEXTUALIZER_DISABLED_ALIAS,
    CONTEXTUALIZER_PROVIDER_ORDER,
    DEFAULT_ANTHROPIC_CONTEXTUALIZER_MODEL,
    NOOP_CONTEXTUALIZER_ID,
)
from rag_core.documents.ocr_provider_names import (
    COMMAND_OCR_PROVIDER,
    DEFAULT_GEMINI_OCR_MODEL,
    DEFAULT_MISTRAL_OCR_MODEL,
    GEMINI_OCR_PROVIDER,
    MISTRAL_OCR_PROVIDER,
    OCR_PROVIDER_ORDER,
)

from tests.support.source_graph import import_graph, iter_package_sources

# Every layer that consumes provider names; ownership is asserted across this
# union so a provider id cannot be redeclared closer to a call site.
PROVIDER_NAME_ROOTS = (
    "src/rag_core/documents",
    "src/rag_core/search",
    "src/rag_core/cli",
    "src/rag_core/_engine",
    "src/rag_core/config",
)


def _modules_defining(name: str, *roots: str) -> set[str]:
    """Dotted modules under ``roots`` with a top-level def/class/assign of ``name``."""
    owners: set[str] = set()
    for _rel, dotted, source in iter_package_sources(*roots):
        for node in ast.iter_child_nodes(ast.parse(source)):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                if node.name == name:
                    owners.add(dotted)
            elif isinstance(node, ast.Assign):
                if any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                    owners.add(dotted)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id == name:
                    owners.add(dotted)
    return owners


def _imports_owner(dotted_module: str, owner: str) -> bool:
    graph = import_graph(*PROVIDER_NAME_ROOTS)
    imports = graph.get(dotted_module, set())
    return any(
        imported == owner or imported.startswith(f"{owner}.") for imported in imports
    )


def test_ocr_provider_names_have_single_owner() -> None:
    owner = "rag_core.documents.ocr_provider_names"

    assert MISTRAL_OCR_PROVIDER == "mistral"
    assert GEMINI_OCR_PROVIDER == "gemini"
    assert COMMAND_OCR_PROVIDER == "command"
    assert OCR_PROVIDER_ORDER == ("mistral", "gemini")
    assert DEFAULT_MISTRAL_OCR_MODEL == "mistral-ocr-latest"
    assert DEFAULT_GEMINI_OCR_MODEL == "gemini-2.5-flash"

    for constant in (
        "MISTRAL_OCR_PROVIDER",
        "GEMINI_OCR_PROVIDER",
        "COMMAND_OCR_PROVIDER",
        "OCR_PROVIDER_ORDER",
        "DEFAULT_MISTRAL_OCR_MODEL",
        "DEFAULT_GEMINI_OCR_MODEL",
    ):
        assert _modules_defining(constant, *PROVIDER_NAME_ROOTS) == {owner}

    # The diagnostics and CLI doctor surfaces reach the names through the owner
    # rather than re-listing provider ids/models.
    assert _imports_owner(
        "rag_core.search.providers.provider_diagnostics", owner
    )
    assert _imports_owner("rag_core.cli.doctor_output", owner)


def test_contextualizer_provider_names_have_single_owner() -> None:
    owner = "rag_core.documents.contextualizer_provider_names"

    assert NOOP_CONTEXTUALIZER_ID == "noop"
    assert CONTEXTUALIZER_DISABLED_ALIAS == "none"
    assert ANTHROPIC_CONTEXTUALIZER_ID == "anthropic"
    assert CONTEXTUALIZER_PROVIDER_ORDER == ("noop", "anthropic")
    assert DEFAULT_ANTHROPIC_CONTEXTUALIZER_MODEL == "claude-haiku-4-5-20251001"

    for constant in (
        "NOOP_CONTEXTUALIZER_ID",
        "CONTEXTUALIZER_DISABLED_ALIAS",
        "ANTHROPIC_CONTEXTUALIZER_ID",
        "CONTEXTUALIZER_PROVIDER_ORDER",
        "DEFAULT_ANTHROPIC_CONTEXTUALIZER_MODEL",
    ):
        assert _modules_defining(constant, *PROVIDER_NAME_ROOTS) == {owner}

    # The anthropic default model and id flow into the adapter from the owner,
    # not a private re-declaration.
    assert _imports_owner("rag_core.documents.contextualizer_adapters", owner)
    assert _imports_owner("rag_core._engine.core_runtime", owner)
    assert _imports_owner(
        "rag_core.search.providers.provider_diagnostics", owner
    )
    assert _imports_owner("rag_core.cli.doctor_output", owner)
