from __future__ import annotations

from pathlib import Path

from rag_core.provider_api_keys import (
    ANTHROPIC_API_KEY_ENVS,
    ANTHROPIC_API_PROVIDER,
    COHERE_API_KEY_ENVS,
    GEMINI_API_KEY_ENVS,
    GEMINI_API_PROVIDER,
    GOOGLE_API_PROVIDER_ALIAS,
    GOOGLE_API_KEY_ENVS,
    MISTRAL_API_KEY_ENVS,
    MISTRAL_API_PROVIDER,
    OPENAI_API_KEY_ENVS,
    PROVIDER_API_KEY_ENVS,
    PROVIDER_NAMES_WITH_API_KEYS,
    QDRANT_API_KEY_ENVS,
    TURBOPUFFER_API_KEY_ENVS,
    VOYAGE_API_KEY_ENVS,
    VOYAGE_API_PACKAGE_ALIAS,
    ZEROENTROPY_API_KEY_ENVS,
    all_provider_api_key_env_names,
    api_key_configured,
    first_configured_api_key,
    normalize_api_key,
    provider_api_key_env_names,
)
from rag_core.provider_package_names import (
    ANTHROPIC_PACKAGE,
    COHERE_PACKAGE,
    FASTEMBED_PACKAGE,
    GEMINI_PACKAGE,
    GOOGLE_PACKAGE_ALIAS as GOOGLE_PACKAGE_NAME_ALIAS,
    MISTRAL_PACKAGE,
    OPENAI_PACKAGE,
    OPENTELEMETRY_TRACE_PACKAGE,
    PROVIDER_ERROR_MODULES,
    VOYAGE_PACKAGE,
    ZEROENTROPY_PACKAGE,
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


def test_provider_api_key_env_names_have_single_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/provider_api_keys.py",
            "src/rag_core/cli_provider_errors.py",
            "src/rag_core/documents/ocr_command_runtime.py",
            "src/rag_core/documents/ocr_commands/mistral.py",
            "src/rag_core/documents/ocr_commands/gemini.py",
            "src/rag_core/cli_config_parser.py",
            "src/rag_core/search/providers/provider_category_diagnostics.py",
            "src/rag_core/search/providers/reranker_resolution.py",
            "src/rag_core/search/providers/model_provider_diagnostics.py",
            "src/rag_core/search/providers/vector_store_diagnostics.py",
            "src/rag_core/search/providers/provider_category_helpers.py",
        )
    }

    assert ANTHROPIC_API_KEY_ENVS == ("ANTHROPIC_API_KEY",)
    assert COHERE_API_KEY_ENVS == ("COHERE_API_KEY", "CO_API_KEY")
    assert GEMINI_API_KEY_ENVS == ("GOOGLE_API_KEY", "GEMINI_API_KEY")
    assert GOOGLE_API_KEY_ENVS == GEMINI_API_KEY_ENVS
    assert MISTRAL_API_KEY_ENVS == ("MISTRAL_API_KEY",)
    assert OPENAI_API_KEY_ENVS == ("OPENAI_API_KEY",)
    assert QDRANT_API_KEY_ENVS == ("RAG_CORE_QDRANT_API_KEY",)
    assert TURBOPUFFER_API_KEY_ENVS == ("TURBOPUFFER_API_KEY",)
    assert VOYAGE_API_KEY_ENVS == ("VOYAGE_API_KEY",)
    assert ZEROENTROPY_API_KEY_ENVS == ("ZEROENTROPY_API_KEY",)
    assert ANTHROPIC_API_PROVIDER == "anthropic"
    assert GEMINI_API_PROVIDER == "gemini"
    assert GOOGLE_API_PROVIDER_ALIAS == "google"
    assert MISTRAL_API_PROVIDER == "mistral"
    assert VOYAGE_API_PACKAGE_ALIAS == "voyageai"
    assert PROVIDER_API_KEY_ENVS[GOOGLE_API_PROVIDER_ALIAS] == GEMINI_API_KEY_ENVS
    assert PROVIDER_API_KEY_ENVS[VOYAGE_API_PACKAGE_ALIAS] == VOYAGE_API_KEY_ENVS
    assert PROVIDER_NAMES_WITH_API_KEYS == tuple(PROVIDER_API_KEY_ENVS)
    assert provider_api_key_env_names("gemini") == GEMINI_API_KEY_ENVS
    assert provider_api_key_env_names("unknown") == ()
    assert normalize_api_key(" secret ") == "secret"
    assert normalize_api_key(None) == ""
    assert (
        first_configured_api_key(("ONE", "TWO"), get_env={"TWO": " two "}.get) == "two"
    )
    assert (
        first_configured_api_key(
            ("ONE",),
            explicit_key=" explicit ",
            get_env={"ONE": "env"}.get,
        )
        == "explicit"
    )
    assert api_key_configured(("ONE",), get_env={"ONE": " env "}.get) is True
    assert api_key_configured(("ONE",), get_env=lambda _name: None) is False
    assert all_provider_api_key_env_names() == (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "COHERE_API_KEY",
        "CO_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "MISTRAL_API_KEY",
        "VOYAGE_API_KEY",
        "ZEROENTROPY_API_KEY",
    )

    owner = sources["src/rag_core/provider_api_keys.py"]
    assert owner.count('ANTHROPIC_API_KEY_ENVS = ("ANTHROPIC_API_KEY",)') == 1
    assert owner.count('COHERE_API_KEY_ENVS = ("COHERE_API_KEY", "CO_API_KEY")') == 1
    assert (
        owner.count('GEMINI_API_KEY_ENVS = ("GOOGLE_API_KEY", "GEMINI_API_KEY")') == 1
    )
    assert owner.count('MISTRAL_API_KEY_ENVS = ("MISTRAL_API_KEY",)') == 1
    assert owner.count('OPENAI_API_KEY_ENVS = ("OPENAI_API_KEY",)') == 1
    assert owner.count('QDRANT_API_KEY_ENVS = ("RAG_CORE_QDRANT_API_KEY",)') == 1
    assert owner.count('TURBOPUFFER_API_KEY_ENVS = ("TURBOPUFFER_API_KEY",)') == 1
    assert owner.count('VOYAGE_API_KEY_ENVS = ("VOYAGE_API_KEY",)') == 1
    assert owner.count('ZEROENTROPY_API_KEY_ENVS = ("ZEROENTROPY_API_KEY",)') == 1
    assert owner.count('ANTHROPIC_API_PROVIDER = "anthropic"') == 1
    assert owner.count('GEMINI_API_PROVIDER = "gemini"') == 1
    assert owner.count('GOOGLE_API_PROVIDER_ALIAS = "google"') == 1
    assert owner.count('MISTRAL_API_PROVIDER = "mistral"') == 1
    assert owner.count("VOYAGE_API_PACKAGE_ALIAS = VOYAGE_PACKAGE") == 1
    for symbol in (
        "ANTHROPIC_API_PROVIDER",
        "DEFAULT_EMBEDDING_PROVIDER",
        "COHERE_RERANKER_PROVIDER",
        "GEMINI_API_PROVIDER",
        "MISTRAL_API_PROVIDER",
        "VOYAGE_PROVIDER",
        "ZEROENTROPY_PROVIDER",
    ):
        assert symbol in owner
    for duplicate in (
        '"openai"',
        '"cohere"',
        '"voyage"',
        '"zeroentropy"',
    ):
        assert duplicate not in owner
    assert owner.count("def normalize_api_key(") == 1
    assert owner.count("def first_configured_api_key(") == 1
    assert owner.count("def api_key_configured(") == 1

    consumers = "\n".join(
        source
        for path, source in sources.items()
        if path != "src/rag_core/provider_api_keys.py"
    )
    assert "PROVIDER_NAMES_WITH_API_KEYS" in consumers
    assert "provider_api_key_env_names" in consumers
    assert "GEMINI_API_KEY_ENVS" in consumers
    assert "MISTRAL_API_KEY_ENVS" in consumers
    assert "OPENAI_API_KEY_ENVS" in consumers
    assert "QDRANT_API_KEY_ENVS" in consumers
    assert "TURBOPUFFER_API_KEY_ENVS" in consumers
    assert "COHERE_API_KEY_ENVS" in consumers
    assert "_PROVIDER_ENV_VARS = {" not in consumers
    assert "_GEMINI_API_KEY_ENVS" not in consumers
    assert "_API_KEY_ENV_BY_PROVIDER" not in consumers
    assert "def _normalize_optional_str" not in consumers
    assert "def _first_configured_api_key" not in consumers
    assert "def _api_env_configured" not in consumers
    assert 'os.environ.get(env_name, "").strip()' not in consumers
    assert 'bool((explicit_key or "").strip())' not in consumers
    for duplicate in (
        '("ANTHROPIC_API_KEY",)',
        '("OPENAI_API_KEY",)',
        '("COHERE_API_KEY", "CO_API_KEY")',
        '("GOOGLE_API_KEY", "GEMINI_API_KEY")',
        '("GEMINI_API_KEY", "GOOGLE_API_KEY")',
        '("MISTRAL_API_KEY",)',
        '("VOYAGE_API_KEY",)',
        '("ZEROENTROPY_API_KEY",)',
        '("RAG_CORE_QDRANT_API_KEY",)',
        '("TURBOPUFFER_API_KEY",)',
    ):
        assert duplicate not in consumers





def test_provider_package_names_have_single_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/provider_package_names.py",
            "src/rag_core/provider_api_keys.py",
            "src/rag_core/cli_provider_errors.py",
            "src/rag_core/search/providers/provider_category_diagnostics.py",
            "src/rag_core/search/providers/model_provider_diagnostics.py",
            "src/rag_core/search/providers/event_sink_category_diagnostics.py",
        )
    }

    assert ANTHROPIC_PACKAGE == "anthropic"
    assert COHERE_PACKAGE == "cohere"
    assert FASTEMBED_PACKAGE == "fastembed"
    assert GEMINI_PACKAGE == "gemini"
    assert GOOGLE_PACKAGE_NAME_ALIAS == "google"
    assert MISTRAL_PACKAGE == "mistral"
    assert OPENAI_PACKAGE == "openai"
    assert OPENTELEMETRY_TRACE_PACKAGE == "opentelemetry.trace"
    assert VOYAGE_PACKAGE == "voyageai"
    assert ZEROENTROPY_PACKAGE == "zeroentropy"
    assert PROVIDER_ERROR_MODULES == (
        ANTHROPIC_PACKAGE,
        COHERE_PACKAGE,
        GEMINI_PACKAGE,
        GOOGLE_PACKAGE_NAME_ALIAS,
        MISTRAL_PACKAGE,
        OPENAI_PACKAGE,
        VOYAGE_PACKAGE,
        ZEROENTROPY_PACKAGE,
    )

    owner = sources["src/rag_core/provider_package_names.py"]
    for definition in (
        'ANTHROPIC_PACKAGE = "anthropic"',
        'COHERE_PACKAGE = "cohere"',
        'FASTEMBED_PACKAGE = "fastembed"',
        'GEMINI_PACKAGE = "gemini"',
        'GOOGLE_PACKAGE_ALIAS = "google"',
        'MISTRAL_PACKAGE = "mistral"',
        'OPENAI_PACKAGE = "openai"',
        'OPENTELEMETRY_TRACE_PACKAGE = "opentelemetry.trace"',
        'VOYAGE_PACKAGE = "voyageai"',
        'ZEROENTROPY_PACKAGE = "zeroentropy"',
        "PROVIDER_ERROR_MODULES = (",
    ):
        assert owner.count(definition) == 1

    consumers = "\n".join(
        source
        for path, source in sources.items()
        if path != "src/rag_core/provider_package_names.py"
    )
    for symbol in (
        "ANTHROPIC_PACKAGE",
        "COHERE_PACKAGE",
        "FASTEMBED_PACKAGE",
        "GEMINI_PACKAGE",
        "GOOGLE_PACKAGE_ALIAS",
        "MISTRAL_PACKAGE",
        "OPENAI_PACKAGE",
        "OPENTELEMETRY_TRACE_PACKAGE",
        "PROVIDER_ERROR_MODULES",
        "VOYAGE_PACKAGE",
        "ZEROENTROPY_PACKAGE",
    ):
        assert symbol in consumers
    for duplicate in (
        "_PROVIDER_MODULES = {",
        'package_name="openai"',
        'package_name="cohere"',
        'package_name="voyageai"',
        'package_name="zeroentropy"',
        '"fastembed"',
        '"opentelemetry.trace"',
    ):
        assert duplicate not in consumers
