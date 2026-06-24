from __future__ import annotations

from rag_core.config import (
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_RERANKER_PROVIDER,
    DEMO_EMBEDDING_PROVIDER,
    LOCAL_EMBEDDING_PROVIDER,
)
from rag_core.search.providers.cohere import (
    COHERE_PROVIDER,
    DEFAULT_COHERE_EMBEDDING_MODEL,
    DEFAULT_COHERE_RERANKER_MODEL,
)
from rag_core.search.providers.provider_diagnostics import (
    EMBEDDING_PROVIDER_ORDER,
    RERANKER_PROVIDER_ORDER,
)
from rag_core.search.providers.voyage import (
    DEFAULT_VOYAGE_EMBEDDING_MODEL,
    DEFAULT_VOYAGE_RERANKER_MODEL,
    VOYAGE_PROVIDER,
)
from rag_core.search.providers.zeroentropy import (
    DEFAULT_ZEROENTROPY_EMBEDDING_MODEL,
    DEFAULT_ZEROENTROPY_RERANKER_MODEL,
    ZEROENTROPY_PROVIDER,
)

from tests.support.source_graph import defining_modules

SRC = "src/rag_core"


def test_model_provider_ids_have_single_adapter_or_config_owners() -> None:
    assert DEFAULT_EMBEDDING_PROVIDER == "openai"
    assert DEMO_EMBEDDING_PROVIDER == "demo"
    assert LOCAL_EMBEDDING_PROVIDER == "local"
    assert DEFAULT_RERANKER_PROVIDER == "none"
    assert COHERE_PROVIDER == "cohere"
    assert VOYAGE_PROVIDER == "voyage"
    assert ZEROENTROPY_PROVIDER == "zeroentropy"
    assert EMBEDDING_PROVIDER_ORDER == (
        DEFAULT_EMBEDDING_PROVIDER,
        DEMO_EMBEDDING_PROVIDER,
        LOCAL_EMBEDDING_PROVIDER,
        COHERE_PROVIDER,
        VOYAGE_PROVIDER,
        ZEROENTROPY_PROVIDER,
    )
    assert RERANKER_PROVIDER_ORDER == (
        DEFAULT_RERANKER_PROVIDER,
        COHERE_PROVIDER,
        VOYAGE_PROVIDER,
        ZEROENTROPY_PROVIDER,
    )

    # Each provider id and the canonical provider-order tuples have exactly one
    # owning module. Asserting on the import graph (where the symbol is bound)
    # rather than a hand-pinned source-string list survives file merges/renames
    # while still rejecting a duplicate definition anywhere under src/.
    owners = {
        "DEFAULT_EMBEDDING_PROVIDER": "rag_core.config.embedding_config",
        "DEMO_EMBEDDING_PROVIDER": "rag_core.config.embedding_config",
        "LOCAL_EMBEDDING_PROVIDER": "rag_core.config.embedding_config",
        "DEFAULT_RERANKER_PROVIDER": "rag_core.config.reranker_config",
        "COHERE_PROVIDER": "rag_core.search.providers.cohere",
        "VOYAGE_PROVIDER": "rag_core.search.providers.voyage",
        "ZEROENTROPY_PROVIDER": "rag_core.search.providers.zeroentropy",
        "EMBEDDING_PROVIDER_ORDER": "rag_core.search.providers.provider_diagnostics",
        "RERANKER_PROVIDER_ORDER": "rag_core.search.providers.provider_diagnostics",
    }
    for name, owner in owners.items():
        assert defining_modules(SRC, name=name) == {owner}


def test_embedding_provider_default_models_have_provider_module_owners() -> None:
    assert DEFAULT_COHERE_EMBEDDING_MODEL == "embed-v4.0"
    assert DEFAULT_VOYAGE_EMBEDDING_MODEL == "voyage-4"
    assert DEFAULT_ZEROENTROPY_EMBEDDING_MODEL == "zembed-1"

    # Each default embedding-model constant lives in its provider adapter and
    # nowhere else; the embedding factory must reference the owner, not inline a
    # second copy of the value.
    assert defining_modules(SRC, name="DEFAULT_COHERE_EMBEDDING_MODEL") == {
        "rag_core.search.providers.cohere"
    }
    assert defining_modules(SRC, name="DEFAULT_VOYAGE_EMBEDDING_MODEL") == {
        "rag_core.search.providers.voyage"
    }
    assert defining_modules(SRC, name="DEFAULT_ZEROENTROPY_EMBEDDING_MODEL") == {
        "rag_core.search.providers.zeroentropy"
    }


def test_reranker_default_models_have_provider_module_owners() -> None:
    assert DEFAULT_COHERE_RERANKER_MODEL == "rerank-v3.5"
    assert DEFAULT_VOYAGE_RERANKER_MODEL == "rerank-2.5-lite"
    assert DEFAULT_ZEROENTROPY_RERANKER_MODEL == "zerank-2"

    # Each default reranker-model constant lives in its provider adapter and
    # nowhere else; the reranker factory must reference the owner, not inline a
    # second copy of the value.
    assert defining_modules(SRC, name="DEFAULT_COHERE_RERANKER_MODEL") == {
        "rag_core.search.providers.cohere"
    }
    assert defining_modules(SRC, name="DEFAULT_VOYAGE_RERANKER_MODEL") == {
        "rag_core.search.providers.voyage"
    }
    assert defining_modules(SRC, name="DEFAULT_ZEROENTROPY_RERANKER_MODEL") == {
        "rag_core.search.providers.zeroentropy"
    }
