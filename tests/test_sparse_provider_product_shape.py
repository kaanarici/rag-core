from __future__ import annotations

from rag_core.config.vector_store_config import (
    DEFAULT_TURBOPUFFER_DELETE_CONTINUATION_LIMIT,
)
from rag_core.search.providers.diagnostic_support import (
    PROVIDER_DIAGNOSTIC_READINESS_SCOPES,
    READINESS_INSTALLED_AND_CONFIGURED,
)
from rag_core.search.providers.sparse import (
    DEFAULT_SPARSE_EMBEDDER_PROVIDER,
    SPARSE_EMBEDDER_PROVIDER_ORDER,
    SPARSE_LOAD_DISABLED,
    SPARSE_LOAD_FAILED,
    SPARSE_LOAD_LOADED,
    SPARSE_LOAD_NOT_CHECKED_BY_DOCTOR,
    SPARSE_LOAD_NOT_LOADED,
    SPLADE_LOAD_UNKNOWN_UNTIL_RUN,
    FastEmbedSparseEmbedder,
)
from rag_core.search.providers.turbopuffer_client import (
    validate_turbopuffer_delete_continuation_limit,
)

from tests.support.source_graph import (
    defining_modules,
    modules_importing,
    symbol_module,
    under_module,
)

SRC = "src/rag_core"
SPARSE_OWNER = "rag_core.search.providers.sparse"
SUPPORT_OWNER = "rag_core.search.providers.diagnostic_support"


def test_sparse_provider_default_has_single_provider_owner() -> None:
    assert DEFAULT_SPARSE_EMBEDDER_PROVIDER == "fastembed"
    assert SPARSE_EMBEDDER_PROVIDER_ORDER == (DEFAULT_SPARSE_EMBEDDER_PROVIDER,)
    assert READINESS_INSTALLED_AND_CONFIGURED == "installed_and_configured"
    assert PROVIDER_DIAGNOSTIC_READINESS_SCOPES == (READINESS_INSTALLED_AND_CONFIGURED,)

    # The sparse embedder class derives its provider_name from the default
    # constant, so the constant is the single source of the id; the readiness
    # scope value lives only in diagnostic_support. Assert ownership via the
    # import graph rather than scraping each owner's source line.
    assert FastEmbedSparseEmbedder.provider_name == DEFAULT_SPARSE_EMBEDDER_PROVIDER
    assert symbol_module(FastEmbedSparseEmbedder) == SPARSE_OWNER
    assert defining_modules(SRC, name="DEFAULT_SPARSE_EMBEDDER_PROVIDER") == {
        SPARSE_OWNER
    }
    assert defining_modules(SRC, name="SPARSE_EMBEDDER_PROVIDER_ORDER") == {SPARSE_OWNER}
    assert defining_modules(SRC, name="READINESS_INSTALLED_AND_CONFIGURED") == {
        SUPPORT_OWNER
    }
    assert defining_modules(SRC, name="PROVIDER_DIAGNOSTIC_READINESS_SCOPES") == {
        SUPPORT_OWNER
    }


def test_sparse_provider_load_status_labels_have_single_owner() -> None:
    assert SPARSE_LOAD_NOT_LOADED == "not_loaded"
    assert SPARSE_LOAD_DISABLED == "disabled"
    assert SPARSE_LOAD_LOADED == "loaded"
    assert SPARSE_LOAD_FAILED == "load_failed"
    assert SPARSE_LOAD_NOT_CHECKED_BY_DOCTOR == "not_checked_by_doctor"
    assert SPLADE_LOAD_UNKNOWN_UNTIL_RUN == "unknown_until_sparse_embedding_runs"

    # Every sparse load-status label has a single owner module; nothing else
    # under src/ may bind the same name, so consumers reference the one constant
    # instead of re-spelling the literal.
    for name in (
        "SPARSE_LOAD_NOT_LOADED",
        "SPARSE_LOAD_DISABLED",
        "SPARSE_LOAD_LOADED",
        "SPARSE_LOAD_FAILED",
        "SPARSE_LOAD_NOT_CHECKED_BY_DOCTOR",
        "SPLADE_LOAD_UNKNOWN_UNTIL_RUN",
    ):
        assert defining_modules(SRC, name=name) == {SPARSE_OWNER}


def test_turbopuffer_delete_continuation_default_has_single_source() -> None:
    assert DEFAULT_TURBOPUFFER_DELETE_CONTINUATION_LIMIT == 1_000

    # The numeric default has one owner module; the turbopuffer write path must
    # delegate to the shared validator rather than inline its own bound, so the
    # value is never duplicated as a second literal.
    assert defining_modules(SRC, name="DEFAULT_TURBOPUFFER_DELETE_CONTINUATION_LIMIT") == {
        "rag_core.config.vector_store_config"
    }
    assert symbol_module(validate_turbopuffer_delete_continuation_limit) == (
        "rag_core.search.providers.turbopuffer_client"
    )
    write_imports = modules_importing(
        "src/rag_core/search/providers",
        predicate=under_module(
            "rag_core.search.providers.turbopuffer_client"
            ".validate_turbopuffer_delete_continuation_limit"
        ),
    )
    assert "rag_core.search.providers.turbopuffer_store" in write_imports
