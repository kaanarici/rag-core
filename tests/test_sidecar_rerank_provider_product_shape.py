from __future__ import annotations

import ast
from pathlib import Path

from rag_core.retrieval_defaults import (
    DEFAULT_SEARCH_LIMIT,
)
from rag_core.search.lexical_sidecar import (
    PORTABLE_LEXICAL_SIDECAR_PROVIDER,
    SEARCH_SIDECAR_PROVIDER_ORDER,
    PortableLexicalSidecar,
)

from tests.support.source_graph import defining_modules, symbol_module

SRC = "src/rag_core"
SIDECAR_OWNER = "rag_core.search.lexical_sidecar"


def test_search_sidecar_provider_defaults_have_single_sidecar_owner() -> None:
    assert PORTABLE_LEXICAL_SIDECAR_PROVIDER == "portable_lexical"
    assert SEARCH_SIDECAR_PROVIDER_ORDER == ("portable_lexical",)
    # The provider id is derived from the sidecar class, so the class is the real
    # owner of the value; assert that linkage plus single-ownership of both the
    # id and the order tuple, rather than scraping the owner's source line.
    assert PortableLexicalSidecar.provider_name == PORTABLE_LEXICAL_SIDECAR_PROVIDER
    assert symbol_module(PortableLexicalSidecar) == SIDECAR_OWNER
    assert defining_modules(SRC, name="PORTABLE_LEXICAL_SIDECAR_PROVIDER") == {
        SIDECAR_OWNER
    }
    assert defining_modules(SRC, name="SEARCH_SIDECAR_PROVIDER_ORDER") == {SIDECAR_OWNER}





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
