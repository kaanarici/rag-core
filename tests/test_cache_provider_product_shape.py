from __future__ import annotations

from rag_core.search.providers.cache_sqlite import (
    CACHE_PROVIDER_ORDER,
    IN_MEMORY_CACHE_PROVIDER,
    NO_CACHE_PROVIDER,
    SQLITE_CACHE_PROVIDER,
)
from rag_core.search.providers.cached_embedding import DEFAULT_EMBEDDING_NORMALIZATION
from rag_core.search.providers.cached_embedding import (
    EMBEDDING_OPERATION_QUERY,
    EMBEDDING_OPERATION_TEXTS,
)
from rag_core.search.providers.embedding_cache import DEFAULT_EMBEDDING_CACHE_PROVIDER
from rag_core.search.providers.embedding_input_types import (
    EMBEDDING_INPUT_DOCUMENT,
    EMBEDDING_INPUT_QUERY,
)

from tests.support.source_graph import (
    defining_modules,
    import_graph,
    modules_assigning_value,
)

CACHE_PROVIDER_NAMES = "rag_core.search.providers.cache_sqlite"
EMBEDDING_CACHE = "rag_core.search.providers.embedding_cache"
EMBEDDING_INPUT_TYPES = "rag_core.search.providers.embedding_input_types"
CACHED_EMBEDDING = "rag_core.search.providers.cached_embedding"


def test_cache_provider_defaults_have_single_factory_owner() -> None:
    assert DEFAULT_EMBEDDING_CACHE_PROVIDER == "none"
    assert NO_CACHE_PROVIDER == "none"
    assert IN_MEMORY_CACHE_PROVIDER == "in_memory"
    assert SQLITE_CACHE_PROVIDER == "sqlite"
    assert CACHE_PROVIDER_ORDER == (
        NO_CACHE_PROVIDER,
        IN_MEMORY_CACHE_PROVIDER,
        SQLITE_CACHE_PROVIDER,
    )

    # cache_sqlite owns each cache provider literal: no other module
    # under the package re-hardcodes "in_memory"/"sqlite", and the order tuple
    # plus NO_CACHE_PROVIDER live only there.
    assert modules_assigning_value("src/rag_core", value="in_memory") == {
        CACHE_PROVIDER_NAMES: ["IN_MEMORY_CACHE_PROVIDER"]
    }
    assert modules_assigning_value("src/rag_core", value="sqlite") == {
        CACHE_PROVIDER_NAMES: ["SQLITE_CACHE_PROVIDER"]
    }
    assert defining_modules("src/rag_core", name="NO_CACHE_PROVIDER") == {
        CACHE_PROVIDER_NAMES
    }
    assert defining_modules("src/rag_core", name="CACHE_PROVIDER_ORDER") == {
        CACHE_PROVIDER_NAMES
    }

    # The provider order is a registry concept that belongs to the names module;
    # the embedding_cache factory must consume the named default, not re-derive
    # the order or a second DEFAULT_CACHE_PROVIDER.
    assert defining_modules("src/rag_core", name="DEFAULT_EMBEDDING_CACHE_PROVIDER") == {
        EMBEDDING_CACHE
    }
    assert defining_modules("src/rag_core", name="CACHE_PROVIDER_ORDER") != {
        EMBEDDING_CACHE
    }
    assert defining_modules("src/rag_core", name="DEFAULT_CACHE_PROVIDER") == set()

    graph = import_graph(
        "src/rag_core/search/providers",
        "src/rag_core/cli",
        "src/rag_core/_engine",
    )

    def imports_from(module: str, owner: str) -> set[str]:
        return {i for i in graph.get(module, set()) if i.startswith(f"{owner}.")}

    # The diagnostics layer and the doctor CLI both pull the cache order from the
    # names owner instead of importing it through the embedding_cache factory or
    # re-listing providers inline.
    diagnostics = "rag_core.search.providers.provider_diagnostics"
    assert f"{CACHE_PROVIDER_NAMES}.CACHE_PROVIDER_ORDER" in graph[diagnostics]
    assert f"{CACHE_PROVIDER_NAMES}.NO_CACHE_PROVIDER" in graph[diagnostics]
    assert imports_from(diagnostics, EMBEDDING_CACHE) == set()
    assert f"{CACHE_PROVIDER_NAMES}.CACHE_PROVIDER_ORDER" in graph["rag_core.cli.doctor_output"]

    # The assembly path references the named sqlite provider rather than a bare
    # "sqlite" literal: cache_sqlite is the only assigner of that value.
    assembly = "rag_core._engine.core_assembly"
    assert f"{CACHE_PROVIDER_NAMES}.SQLITE_CACHE_PROVIDER" in graph[assembly]


def test_embedding_input_types_have_single_provider_owner() -> None:
    assert EMBEDDING_INPUT_DOCUMENT == "document"
    assert EMBEDDING_INPUT_QUERY == "query"

    # embedding_input_types is the sole assigner of the "document"/"query" input
    # literals; every consumer (cached_embedding, voyage, zeroentropy, tests)
    # must reference the named constants rather than re-hardcoding the strings.
    assert modules_assigning_value("src/rag_core", value="document") == {
        EMBEDDING_INPUT_TYPES: ["EMBEDDING_INPUT_DOCUMENT"]
    }
    assert modules_assigning_value("src/rag_core", value="query") == {
        EMBEDDING_INPUT_TYPES: ["EMBEDDING_INPUT_QUERY"]
    }


def test_cached_embedding_operation_labels_have_single_owner() -> None:
    assert EMBEDDING_OPERATION_TEXTS == "embed_texts"
    assert EMBEDDING_OPERATION_QUERY == "embed_query"

    # cached_embedding owns the operation labels; no other module under the
    # package binds the "embed_texts"/"embed_query" literals.
    assert modules_assigning_value("src/rag_core", value="embed_texts") == {
        CACHED_EMBEDDING: ["EMBEDDING_OPERATION_TEXTS"]
    }
    assert modules_assigning_value("src/rag_core", value="embed_query") == {
        CACHED_EMBEDDING: ["EMBEDDING_OPERATION_QUERY"]
    }


def test_cached_embedding_normalization_default_has_single_owner() -> None:
    assert DEFAULT_EMBEDDING_NORMALIZATION == "text_sha256_utf8"

    # The normalization scheme literal lives only on cached_embedding's default.
    assert modules_assigning_value("src/rag_core", value="text_sha256_utf8") == {
        CACHED_EMBEDDING: ["DEFAULT_EMBEDDING_NORMALIZATION"]
    }
