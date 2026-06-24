from __future__ import annotations

from rag_core.provider_api_keys import (
    ANTHROPIC_API_KEY_ENVS,
    ANTHROPIC_API_PROVIDER,
    COHERE_API_KEY_ENVS,
    GEMINI_API_KEY_ENVS,
    GEMINI_API_PROVIDER,
    GOOGLE_API_KEY_ENVS,
    GOOGLE_API_PROVIDER_ALIAS,
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

from tests.support.source_graph import defining_modules, symbol_module

SRC = "src/rag_core"
API_KEYS_OWNER = "rag_core.provider_api_keys"
PACKAGE_NAMES_OWNER = "rag_core.provider_package_names"


def test_provider_api_key_env_names_have_single_owner() -> None:
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

    # Each env-name tuple, provider/alias id, and the api-key helpers have a
    # single owner module; nothing under src/ may define a second copy. The graph
    # check (where each name is bound) replaces the old hand-pinned path list plus
    # the "duplicate literal not in consumers" substring guards.
    for name in (
        "ANTHROPIC_API_KEY_ENVS",
        "COHERE_API_KEY_ENVS",
        "GEMINI_API_KEY_ENVS",
        "MISTRAL_API_KEY_ENVS",
        "OPENAI_API_KEY_ENVS",
        "QDRANT_API_KEY_ENVS",
        "TURBOPUFFER_API_KEY_ENVS",
        "VOYAGE_API_KEY_ENVS",
        "ZEROENTROPY_API_KEY_ENVS",
        "ANTHROPIC_API_PROVIDER",
        "GEMINI_API_PROVIDER",
        "GOOGLE_API_PROVIDER_ALIAS",
        "MISTRAL_API_PROVIDER",
        "VOYAGE_API_PACKAGE_ALIAS",
        "PROVIDER_API_KEY_ENVS",
        "PROVIDER_NAMES_WITH_API_KEYS",
    ):
        assert defining_modules(SRC, name=name) == {API_KEYS_OWNER}

    for func in (
        normalize_api_key,
        first_configured_api_key,
        api_key_configured,
        provider_api_key_env_names,
        all_provider_api_key_env_names,
    ):
        assert symbol_module(func) == API_KEYS_OWNER
    # The pure key helpers must have exactly one definition. provider_api_key_env_names
    # is excluded: provider_health re-exports a thin delegating wrapper of the same
    # name, which is a legitimate second binding, not a duplicated implementation.
    for func in (
        normalize_api_key,
        first_configured_api_key,
        api_key_configured,
        all_provider_api_key_env_names,
    ):
        assert defining_modules(SRC, name=func.__name__) == {API_KEYS_OWNER}


def test_provider_package_names_have_single_owner() -> None:
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

    # Each package-name constant and the error-module tuple have a single owner;
    # nothing else under src/ may bind them, so consumers import the one copy.
    for name in (
        "ANTHROPIC_PACKAGE",
        "COHERE_PACKAGE",
        "FASTEMBED_PACKAGE",
        "GEMINI_PACKAGE",
        "GOOGLE_PACKAGE_ALIAS",
        "MISTRAL_PACKAGE",
        "OPENAI_PACKAGE",
        "OPENTELEMETRY_TRACE_PACKAGE",
        "VOYAGE_PACKAGE",
        "ZEROENTROPY_PACKAGE",
        "PROVIDER_ERROR_MODULES",
    ):
        assert defining_modules(SRC, name=name) == {PACKAGE_NAMES_OWNER}
