from __future__ import annotations

from pathlib import Path

from rag_core.search.providers.cache_provider_names import (
    CACHE_PROVIDER_ORDER,
    IN_MEMORY_CACHE_PROVIDER,
    NO_CACHE_PROVIDER,
    SQLITE_CACHE_PROVIDER,
)
from rag_core.search.providers.cached_embedding import DEFAULT_EMBEDDING_NORMALIZATION
from rag_core.search.providers.cached_embedding_observations import (
    EMBEDDING_OPERATION_QUERY,
    EMBEDDING_OPERATION_TEXTS,
)
from rag_core.search.providers.embedding_cache import DEFAULT_EMBEDDING_CACHE_PROVIDER
from rag_core.search.providers.embedding_input_types import (
    EMBEDDING_INPUT_DOCUMENT,
    EMBEDDING_INPUT_QUERY,
)

CANONICAL_LAUNCH_GATES = (
    "uv run ruff check .",
    "uv run mypy src tests examples",
    "uv run pytest -q",
    "./scripts/dx_smoke.sh",
    "./scripts/ci_self_host_smoke.sh",
    "uv build",
    "uv run python scripts/check_dist_artifacts.py",
    "uv run python scripts/wheel_smoke.py",
)


def test_cache_provider_defaults_have_single_factory_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/search/providers/cache_provider_names.py",
            "src/rag_core/search/providers/embedding_cache.py",
            "src/rag_core/search/providers/embedding_memory_cache.py",
            "src/rag_core/search/providers/chunk_context_cache.py",
            "src/rag_core/search/providers/provider_category_diagnostics.py",
            "src/rag_core/_engine/core_assembly.py",
            "src/rag_core/cli_doctor_output.py",
        )
    }

    assert DEFAULT_EMBEDDING_CACHE_PROVIDER == "none"
    assert NO_CACHE_PROVIDER == "none"
    assert IN_MEMORY_CACHE_PROVIDER == "in_memory"
    assert SQLITE_CACHE_PROVIDER == "sqlite"
    assert CACHE_PROVIDER_ORDER == (
        NO_CACHE_PROVIDER,
        IN_MEMORY_CACHE_PROVIDER,
        SQLITE_CACHE_PROVIDER,
    )
    owner = sources["src/rag_core/search/providers/cache_provider_names.py"]
    assert owner.count('NO_CACHE_PROVIDER = "none"') == 1
    assert owner.count('IN_MEMORY_CACHE_PROVIDER = "in_memory"') == 1
    assert owner.count('SQLITE_CACHE_PROVIDER = "sqlite"') == 1
    assert (
        "CACHE_PROVIDER_ORDER = ("
        in sources["src/rag_core/search/providers/cache_provider_names.py"]
    )
    for source_path in (
        "src/rag_core/search/providers/embedding_cache.py",
        "src/rag_core/search/providers/embedding_memory_cache.py",
        "src/rag_core/search/providers/chunk_context_cache.py",
    ):
        source = sources[source_path]
        assert 'provider_name = "none"' not in source
        assert 'provider_name = "in_memory"' not in source
        assert 'provider_name = "sqlite"' not in source
    assert (
        "DEFAULT_EMBEDDING_CACHE_PROVIDER"
        in sources["src/rag_core/search/providers/embedding_cache.py"]
    )
    assert (
        "CACHE_PROVIDER_ORDER"
        not in sources["src/rag_core/search/providers/embedding_cache.py"]
    )
    assert (
        "DEFAULT_CACHE_PROVIDER"
        not in sources["src/rag_core/search/providers/embedding_cache.py"]
    )
    diagnostics = sources[
        "src/rag_core/search/providers/provider_category_diagnostics.py"
    ]
    assert (
        "from .cache_provider_names import CACHE_PROVIDER_ORDER, NO_CACHE_PROVIDER"
        in diagnostics
    )
    assert "from .embedding_cache import" not in diagnostics
    assert "_CACHE_PROVIDER_ORDER" not in diagnostics
    assert "_CACHE_PROVIDER_ALIASES" not in diagnostics
    assert (
        'default=_normalize(config.ingest.embedding_cache_provider) or "none"'
        not in diagnostics
    )
    assembly = sources["src/rag_core/_engine/core_assembly.py"]
    assert "SQLITE_CACHE_PROVIDER" in assembly
    assert 'embedding_cache_provider == "sqlite"' not in assembly
    assert "embedding_cache_provider='sqlite'" not in assembly
    doctor_output = sources["src/rag_core/cli_doctor_output.py"]
    assert (
        "from rag_core.search.providers.cache_provider_names import CACHE_PROVIDER_ORDER"
        in doctor_output
    )
    assert '("embedding_cache", ("none", "in_memory", "sqlite"))' not in doctor_output
    assert (
        '("chunk_context_cache", ("none", "in_memory", "sqlite"))' not in doctor_output
    )
    assert "(EMBEDDING_CACHE_PROVIDER_CATEGORY, CACHE_PROVIDER_ORDER)" in doctor_output
    assert (
        "(CHUNK_CONTEXT_CACHE_PROVIDER_CATEGORY, CACHE_PROVIDER_ORDER)" in doctor_output
    )





def test_embedding_input_types_have_single_provider_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/search/providers/embedding_input_types.py",
            "src/rag_core/search/providers/cached_embedding.py",
            "src/rag_core/search/providers/cached_embedding_state.py",
            "src/rag_core/search/providers/voyage.py",
            "src/rag_core/search/providers/zeroentropy.py",
            "tests/test_embedding_cache.py",
        )
    }

    assert EMBEDDING_INPUT_DOCUMENT == "document"
    assert EMBEDDING_INPUT_QUERY == "query"
    owner = sources["src/rag_core/search/providers/embedding_input_types.py"]
    assert (
        owner.count('EMBEDDING_INPUT_DOCUMENT: Final[EmbeddingInputType] = "document"')
        == 1
    )
    assert (
        owner.count('EMBEDDING_INPUT_QUERY: Final[EmbeddingInputType] = "query"') == 1
    )

    consumers = "\n".join(
        source
        for path, source in sources.items()
        if path != "src/rag_core/search/providers/embedding_input_types.py"
    )
    assert "EMBEDDING_INPUT_DOCUMENT" in consumers
    assert "EMBEDDING_INPUT_QUERY" in consumers
    for duplicate in (
        'Literal["document", "query"]',
        'input_type="document"',
        'input_type="query"',
        '"input_type": "document"',
        '"input_type": "query"',
    ):
        assert duplicate not in consumers





def test_cached_embedding_operation_labels_have_single_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/search/providers/cached_embedding_observations.py",
            "src/rag_core/search/providers/cached_embedding_runtime.py",
            "src/rag_core/search/providers/cached_embedding_state.py",
            "tests/test_embedding_cache.py",
        )
    }

    assert EMBEDDING_OPERATION_TEXTS == "embed_texts"
    assert EMBEDDING_OPERATION_QUERY == "embed_query"
    owner = sources["src/rag_core/search/providers/cached_embedding_observations.py"]
    assert (
        owner.count(
            'EMBEDDING_OPERATION_TEXTS: Final[EmbeddingOperation] = "embed_texts"'
        )
        == 1
    )
    assert (
        owner.count(
            'EMBEDDING_OPERATION_QUERY: Final[EmbeddingOperation] = "embed_query"'
        )
        == 1
    )

    consumers = "\n".join(
        source
        for path, source in sources.items()
        if path != "src/rag_core/search/providers/cached_embedding_observations.py"
    )
    assert "EMBEDDING_OPERATION_TEXTS" in consumers
    assert "EMBEDDING_OPERATION_QUERY" in consumers
    for duplicate in (
        'operation="embed_texts"',
        'operation="embed_query"',
        'operation == "embed_texts"',
        'operation == "embed_query"',
    ):
        assert duplicate not in consumers





def test_cached_embedding_normalization_default_has_single_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/search/providers/cached_embedding.py",
            "src/rag_core/search/providers/cached_embedding_state.py",
            "tests/test_embedding_cache.py",
        )
    }

    assert DEFAULT_EMBEDDING_NORMALIZATION == "text_sha256_utf8"
    owner = sources["src/rag_core/search/providers/cached_embedding.py"]
    assert owner.count('DEFAULT_EMBEDDING_NORMALIZATION = "text_sha256_utf8"') == 1
    consumers = "\n".join(
        source
        for path, source in sources.items()
        if path != "src/rag_core/search/providers/cached_embedding.py"
    )
    assert "DEFAULT_EMBEDDING_NORMALIZATION" in consumers
    assert '"text_sha256_utf8"' not in consumers
