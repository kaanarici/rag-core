"""Single-owner contracts for the documents runtime helpers.

These assert the *contracts* (owner module, single definition, and that the
documents layer imports the shared helper) via runtime symbol ownership, the
package import graph, and AST definition scans, plus the behavioral value of
each helper. They survive file merges and renames because nothing here pins a
source string, a file path, or a call-site count. (Previously these scraped a
hand-pinned list of source files for substrings and occurrence counts, which
froze the layout and -- for ``safe_http_status`` -- rewarded duplicating
security-sensitive code.)
"""

from __future__ import annotations

import ast

from rag_core.documents.exception_names import exception_type, root_exception_type
from rag_core.documents.http_errors import safe_http_status
from rag_core.documents.page_indices import normalize_page_indices
from rag_core.documents.subprocess_env import (
    COMMON_SUBPROCESS_ENV_KEYS,
    NODE_SUBPROCESS_ENV_KEYS,
    PYTHON_SUBPROCESS_ENV_KEYS,
    TRANSPORT_SUBPROCESS_ENV_KEYS,
)

from tests.support.source_graph import (
    import_graph,
    iter_package_sources,
    modules_importing,
    symbol_module,
    under_module,
)

DOCUMENTS_ROOT = "src/rag_core/documents"


def _modules_defining(root: str, name: str) -> set[str]:
    """Dotted modules under ``root`` with a top-level def/class/assign of ``name``.

    The durable form of "no other file redefines this symbol": it parses the AST
    rather than counting source occurrences, so it survives renames/reformatting
    and -- unlike a call-site count -- never rewards duplication.
    """
    owners: set[str] = set()
    for _rel, dotted, source in iter_package_sources(root):
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


def test_document_subprocess_env_allowlists_have_single_owner() -> None:
    # The subprocess env allowlists are a deliberate single-owner convention:
    # one module defines the tuples so callers cannot quietly re-inline a wider
    # allowlist. Value + single-definition + import-the-owner replace the old
    # path-scraping / raw-key-string negatives.
    assert COMMON_SUBPROCESS_ENV_KEYS == (
        "PATH",
        "HOME",
        "SYSTEMROOT",
        "TMPDIR",
        "TEMP",
        "TMP",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
    )
    assert PYTHON_SUBPROCESS_ENV_KEYS == (
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONUTF8",
        "VIRTUAL_ENV",
    )
    assert NODE_SUBPROCESS_ENV_KEYS == ("NODE_OPTIONS",)
    assert TRANSPORT_SUBPROCESS_ENV_KEYS == (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
    )

    owner = "rag_core.documents.subprocess_env"
    for constant in (
        "COMMON_SUBPROCESS_ENV_KEYS",
        "PYTHON_SUBPROCESS_ENV_KEYS",
        "NODE_SUBPROCESS_ENV_KEYS",
        "TRANSPORT_SUBPROCESS_ENV_KEYS",
    ):
        assert _modules_defining(DOCUMENTS_ROOT, constant) == {owner}

    # The runtime consumers must reach the allowlist through the owner module
    # (and each pulls its own family symbol from it), not maintain their own copy.
    graph = import_graph(DOCUMENTS_ROOT)
    assert (
        f"{owner}.NODE_SUBPROCESS_ENV_KEYS"
        in graph["rag_core.documents.pdf_inspector_runtime"]
    )
    assert (
        f"{owner}.PYTHON_SUBPROCESS_ENV_KEYS"
        in graph["rag_core.documents.ocr_command_runtime"]
    )


def test_ocr_page_index_normalization_has_single_owner() -> None:
    assert normalize_page_indices([2, True, 0, False, 2, -1, "3", 1.0]) == [2, 0]
    assert normalize_page_indices(
        [2, True, 0, False, 2, -1, "3", 1.0],
        sort=True,
    ) == [0, 2]
    assert normalize_page_indices([0, 3, 1], page_count=2) == [0, 1]
    assert normalize_page_indices([], page_count=3, default_all_pages=True) == [0, 1, 2]

    assert symbol_module(normalize_page_indices) == "rag_core.documents.page_indices"
    # Exactly one module defines the normalizer; consumers import it instead of
    # re-implementing the page-index filtering loop.
    assert _modules_defining(DOCUMENTS_ROOT, "normalize_page_indices") == {
        "rag_core.documents.page_indices"
    }
    importers = modules_importing(
        DOCUMENTS_ROOT, predicate=under_module("rag_core.documents.page_indices")
    )
    assert importers != {}


def test_ocr_http_status_sanitization_has_single_owner() -> None:
    class WithStatus:
        code = 429

    class WithBoolStatus:
        code = True

    class WithoutStatus:
        pass

    assert safe_http_status(WithStatus()) == 429
    assert safe_http_status(WithBoolStatus()) == "unknown"
    assert safe_http_status(WithoutStatus()) == "unknown"

    # Security-sensitive sanitization must have exactly one owner. Pinning a
    # call-site COUNT (the prior `== 2`) actively rewarded copies; assert
    # single-definition + import-the-owner instead so the OCR-HTTP dedup is free
    # to land.
    assert symbol_module(safe_http_status) == "rag_core.documents.http_errors"
    assert _modules_defining(DOCUMENTS_ROOT, "safe_http_status") == {
        "rag_core.documents.http_errors"
    }
    assert _modules_defining(DOCUMENTS_ROOT, "_safe_http_status") == set()
    importers = modules_importing(
        DOCUMENTS_ROOT, predicate=under_module("rag_core.documents.http_errors")
    )
    assert importers != {}


def test_document_exception_names_have_single_owner() -> None:
    try:
        try:
            raise OSError("inner")
        except OSError as inner:
            raise ValueError("outer") from inner
    except ValueError as exc:
        assert exception_type(exc) == "ValueError"
        assert root_exception_type(exc) == "OSError"

    assert symbol_module(exception_type) == "rag_core.documents.exception_names"
    assert symbol_module(root_exception_type) == "rag_core.documents.exception_names"
    for name in ("exception_type", "root_exception_type"):
        assert _modules_defining(DOCUMENTS_ROOT, name) == {
            "rag_core.documents.exception_names"
        }
    assert _modules_defining(DOCUMENTS_ROOT, "_exception_type") == set()
    importers = modules_importing(
        DOCUMENTS_ROOT, predicate=under_module("rag_core.documents.exception_names")
    )
    assert importers != {}
