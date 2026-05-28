from __future__ import annotations

from pathlib import Path

from rag_core.config import (
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_RERANKER_PROVIDER,
    DEMO_EMBEDDING_PROVIDER,
)
from rag_core.search.providers.cohere import (
    COHERE_RERANKER_PROVIDER,
    DEFAULT_COHERE_RERANKER_MODEL,
)
from rag_core.search.providers.model_provider_diagnostics import (
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


def test_model_provider_ids_have_single_adapter_or_config_owners() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/config/embedding_config.py",
            "src/rag_core/config/reranker_config.py",
            "src/rag_core/search/providers/cohere.py",
            "src/rag_core/search/providers/voyage.py",
            "src/rag_core/search/providers/zeroentropy.py",
            "src/rag_core/search/providers/embedding.py",
            "src/rag_core/search/providers/reranker.py",
            "src/rag_core/search/providers/reranker_resolution.py",
            "src/rag_core/search/providers/model_provider_diagnostics.py",
            "src/rag_core/cli_doctor_output.py",
        )
    }

    assert DEFAULT_EMBEDDING_PROVIDER == "openai"
    assert DEMO_EMBEDDING_PROVIDER == "demo"
    assert DEFAULT_RERANKER_PROVIDER == "none"
    assert COHERE_RERANKER_PROVIDER == "cohere"
    assert VOYAGE_PROVIDER == "voyage"
    assert ZEROENTROPY_PROVIDER == "zeroentropy"
    assert EMBEDDING_PROVIDER_ORDER == (
        DEFAULT_EMBEDDING_PROVIDER,
        DEMO_EMBEDDING_PROVIDER,
        VOYAGE_PROVIDER,
        ZEROENTROPY_PROVIDER,
    )
    assert RERANKER_PROVIDER_ORDER == (
        DEFAULT_RERANKER_PROVIDER,
        COHERE_RERANKER_PROVIDER,
        VOYAGE_PROVIDER,
        ZEROENTROPY_PROVIDER,
    )
    assert (
        'DEFAULT_EMBEDDING_PROVIDER = "openai"'
        in sources["src/rag_core/config/embedding_config.py"]
    )
    assert (
        'DEMO_EMBEDDING_PROVIDER = "demo"'
        in sources["src/rag_core/config/embedding_config.py"]
    )
    assert (
        'DEFAULT_RERANKER_PROVIDER = "none"'
        in sources["src/rag_core/config/reranker_config.py"]
    )
    assert (
        'COHERE_RERANKER_PROVIDER = "cohere"'
        in sources["src/rag_core/search/providers/cohere.py"]
    )
    assert (
        'VOYAGE_PROVIDER = "voyage"'
        in sources["src/rag_core/search/providers/voyage.py"]
    )
    assert (
        'ZEROENTROPY_PROVIDER = "zeroentropy"'
        in sources["src/rag_core/search/providers/zeroentropy.py"]
    )
    consumers = "\n".join(
        sources[path]
        for path in (
            "src/rag_core/search/providers/embedding.py",
            "src/rag_core/search/providers/reranker.py",
            "src/rag_core/search/providers/reranker_resolution.py",
            "src/rag_core/search/providers/model_provider_diagnostics.py",
            "src/rag_core/cli_doctor_output.py",
        )
    )
    for symbol in (
        "DEFAULT_EMBEDDING_PROVIDER",
        "DEMO_EMBEDDING_PROVIDER",
        "DEFAULT_RERANKER_PROVIDER",
        "COHERE_RERANKER_PROVIDER",
        "VOYAGE_PROVIDER",
        "ZEROENTROPY_PROVIDER",
    ):
        assert symbol in consumers
    assert (
        "EMBEDDING_PROVIDER_ORDER"
        in sources["src/rag_core/search/providers/model_provider_diagnostics.py"]
    )
    assert (
        "RERANKER_PROVIDER_ORDER"
        in sources["src/rag_core/search/providers/model_provider_diagnostics.py"]
    )
    assert "EMBEDDING_PROVIDER_ORDER" in sources["src/rag_core/cli_doctor_output.py"]
    assert "RERANKER_PROVIDER_ORDER" in sources["src/rag_core/cli_doctor_output.py"]
    assert 'EMBEDDING_PROVIDERS.register("openai"' not in consumers
    assert 'EMBEDDING_PROVIDERS.register("demo"' not in consumers
    assert 'EMBEDDING_PROVIDERS.register("voyage"' not in consumers
    assert 'EMBEDDING_PROVIDERS.register("zeroentropy"' not in consumers
    assert 'RERANKER_PROVIDERS.register("none"' not in consumers
    assert 'RERANKER_PROVIDERS.register("cohere"' not in consumers
    assert 'RERANKER_PROVIDERS.register("voyage"' not in consumers
    assert 'RERANKER_PROVIDERS.register("zeroentropy"' not in consumers
    assert '("openai", "voyage", "zeroentropy")' not in consumers
    assert '("none", "cohere", "voyage", "zeroentropy")' not in consumers





def test_embedding_provider_default_models_have_provider_module_owners() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/search/providers/voyage.py",
            "src/rag_core/search/providers/zeroentropy.py",
            "src/rag_core/search/providers/embedding.py",
        )
    }

    assert DEFAULT_VOYAGE_EMBEDDING_MODEL == "voyage-4"
    assert DEFAULT_ZEROENTROPY_EMBEDDING_MODEL == "zembed-1"
    assert (
        sources["src/rag_core/search/providers/voyage.py"].count(
            'DEFAULT_VOYAGE_EMBEDDING_MODEL = "voyage-4"'
        )
        == 1
    )
    assert (
        sources["src/rag_core/search/providers/zeroentropy.py"].count(
            'DEFAULT_ZEROENTROPY_EMBEDDING_MODEL = "zembed-1"'
        )
        == 1
    )
    factory = sources["src/rag_core/search/providers/embedding.py"]
    assert 'model: str = "voyage-4"' not in factory
    assert 'model: str = "zembed-1"' not in factory





def test_reranker_default_models_have_provider_module_owners() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/search/providers/cohere.py",
            "src/rag_core/search/providers/voyage.py",
            "src/rag_core/search/providers/zeroentropy.py",
            "src/rag_core/search/providers/reranker.py",
        )
    }

    assert DEFAULT_COHERE_RERANKER_MODEL == "rerank-v3.5"
    assert DEFAULT_VOYAGE_RERANKER_MODEL == "rerank-2.5-lite"
    assert DEFAULT_ZEROENTROPY_RERANKER_MODEL == "zerank-2"
    assert (
        sources["src/rag_core/search/providers/cohere.py"].count(
            'DEFAULT_COHERE_RERANKER_MODEL = "rerank-v3.5"'
        )
        == 1
    )
    assert (
        sources["src/rag_core/search/providers/voyage.py"].count(
            'DEFAULT_VOYAGE_RERANKER_MODEL = "rerank-2.5-lite"'
        )
        == 1
    )
    assert (
        sources["src/rag_core/search/providers/zeroentropy.py"].count(
            'DEFAULT_ZEROENTROPY_RERANKER_MODEL = "zerank-2"'
        )
        == 1
    )
    factory = sources["src/rag_core/search/providers/reranker.py"]
    assert 'model or "rerank-v3.5"' not in factory
    assert 'model or "rerank-2.5-lite"' not in factory
    assert 'model or "zerank-2"' not in factory
