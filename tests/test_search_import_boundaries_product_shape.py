"""Architectural import-boundary invariants for the search/engine layers.

These check the *contracts* that matter -- layering direction, no imports through
the stale catch-all, and where each search contract is owned -- via the package
import graph and runtime symbol ownership, so the invariants survive file merges,
renames, and reformatting. (Previously these were asserted by scraping a
hand-pinned list of source files for exact import substrings, which froze the
file layout.)
"""

from __future__ import annotations

from rag_core.search.provider_protocols import (
    EmbeddingProvider,
    QueryPlanCapabilities,
    RerankerProvider,
    SearchSidecar,
    SparseEmbedder,
    StoreCapabilities,
)
from rag_core.search.request_models import (
    DeleteFilter,
    RerankBudget,
    SearchQuery,
    StoredDocumentRecord,
)
from rag_core.search.vector_models import (
    SearchResult,
    SparseVector,
    VectorPoint,
)

from tests.support.source_graph import modules_importing, symbol_module, under_module


def test_library_layers_do_not_import_cli_modules() -> None:
    # The library core must never depend on the CLI presentation layer. ingest/
    # is included: the local-search and provider-error helpers it needs live in
    # the neutral rag_core.local_search / rag_core.provider_errors layers, not in
    # cli, so ingest never inverts up into the presentation layer.
    offenders = modules_importing(
        "src/rag_core/_engine",
        "src/rag_core/search",
        "src/rag_core/documents",
        "src/rag_core/ingest",
        "src/rag_core/local_search",
        predicate=under_module("rag_core.cli"),
    )
    assert offenders == {}


def test_search_and_documents_do_not_import_private_engine_modules() -> None:
    # search/ and documents/ sit below _engine in the layering; they must not
    # import up into the private engine package.
    offenders = modules_importing(
        "src/rag_core/search",
        "src/rag_core/documents",
        predicate=under_module("rag_core._engine"),
    )
    assert offenders == {}


def test_ingest_does_not_import_the_private_engine_package() -> None:
    # ingest/ sits below _engine (the engine orchestrates ingest); an ingest ->
    # _engine back-edge forms an _engine<->ingest import cycle. The low-level
    # file/content helpers both layers share live in the neutral rag_core.file_io
    # leaf, so ingest never reaches up into the engine package.
    offenders = modules_importing(
        "src/rag_core/ingest",
        predicate=under_module("rag_core._engine"),
    )
    assert offenders == {}


def test_no_module_imports_through_the_stale_search_catch_all() -> None:
    # rag_core.search.types was a catch-all compatibility layer; nothing may
    # import contracts through it. This guards against re-introducing it, across
    # every module rather than a hand-picked file list.
    offenders = modules_importing(
        "src/rag_core/search",
        "src/rag_core/_engine",
        "src/rag_core/cli",
        predicate=under_module("rag_core.search.types"),
    )
    assert offenders == {}


def test_search_contracts_are_owned_by_their_durable_owner_modules() -> None:
    # Contract ownership is the invariant the per-file "import the owner"
    # assertions were really protecting: each contract must live in one durable
    # owner module so consumers always import it from the same place. Asserting
    # where the symbol is defined (not which files import it) survives merges.
    assert symbol_module(SearchResult) == "rag_core.search.vector_models"
    assert symbol_module(SparseVector) == "rag_core.search.vector_models"
    assert symbol_module(VectorPoint) == "rag_core.search.vector_models"
    assert symbol_module(EmbeddingProvider) == "rag_core.search.provider_protocols"
    assert symbol_module(RerankerProvider) == "rag_core.search.provider_protocols"
    assert symbol_module(SparseEmbedder) == "rag_core.search.provider_protocols"
    assert symbol_module(SearchSidecar) == "rag_core.search.provider_protocols"
    assert symbol_module(StoreCapabilities) == "rag_core.search.provider_protocols"
    assert symbol_module(QueryPlanCapabilities) == "rag_core.search.provider_protocols"
    assert symbol_module(SearchQuery) == "rag_core.search.request_models"
    assert symbol_module(RerankBudget) == "rag_core.search.request_models"
    assert symbol_module(StoredDocumentRecord) == "rag_core.search.request_models"
    assert symbol_module(DeleteFilter) == "rag_core.search.request_models"
