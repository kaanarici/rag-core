import pytest

import rag_core.documents.converters as converters_module
from rag_core.config import DEFAULT_RERANKER_PROVIDER
from rag_core.config.env_access import (
    get_env_bool,
    get_env_float,
    get_env_int,
    get_env_optional_bool,
    get_env_stripped,
)
from rag_core.documents import build_gemini_ocr_provider, build_mistral_ocr_provider
from rag_core.documents.converters import get_converter, registry_loader
from rag_core.documents.converters.registry_specs import ConverterSpec
from rag_core.documents.ocr_provider_names import (
    DEFAULT_GEMINI_OCR_MODEL,
    DEFAULT_MISTRAL_OCR_MODEL,
    GEMINI_OCR_PROVIDER,
    MISTRAL_OCR_PROVIDER,
)
from rag_core.search.providers.reranker import NoOpReranker, create_reranker


def test_get_converter_uses_text_fallbacks() -> None:
    assert get_converter(mime_type="text/plain", filename="notes.bin").format_name == "text"
    assert get_converter(mime_type="application/x-unknown", filename="mystery.bin").format_name == "text"


def test_get_converter_requires_text_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(converters_module, "get_registered_converters", lambda: {})

    with pytest.raises(RuntimeError, match="Text converter is required"):
        converters_module.get_converter(mime_type="application/x-unknown", filename="mystery.bin")


def test_registry_loader_skips_optional_converters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry_loader, "_converters", None)
    monkeypatch.setattr(
        registry_loader,
        "CONVERTER_SPECS",
        (
            ConverterSpec("text", ".text_converter", "TextConverter", required=True),
            ConverterSpec("pdf", ".pdf_converter", "PdfConverter", required=False),
        ),
    )
    monkeypatch.setattr(registry_loader, "REQUIRED_CONVERTER_KEYS", ("text",))

    class _FakeConverter:
        pass

    def _fake_build(spec: ConverterSpec) -> _FakeConverter:
        if spec.key == "text":
            return _FakeConverter()
        raise RuntimeError("optional missing")

    monkeypatch.setattr(registry_loader, "_build_converter", _fake_build)

    converters = registry_loader.get_registered_converters()
    assert set(converters.keys()) == {"text"}


def test_registry_loader_raises_for_required_converter_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry_loader, "_converters", None)
    monkeypatch.setattr(
        registry_loader,
        "CONVERTER_SPECS",
        (ConverterSpec("text", ".text_converter", "TextConverter", required=True),),
    )
    monkeypatch.setattr(registry_loader, "REQUIRED_CONVERTER_KEYS", ("text",))

    def _raise(spec: ConverterSpec) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr(registry_loader, "_build_converter", _raise)

    with pytest.raises(RuntimeError, match="boom"):
        registry_loader.get_registered_converters()


def test_create_reranker_none_has_runtime_metadata() -> None:
    reranker = create_reranker(provider=DEFAULT_RERANKER_PROVIDER)

    assert isinstance(reranker, NoOpReranker)
    assert reranker.provider_name == DEFAULT_RERANKER_PROVIDER
    assert reranker.model_name == DEFAULT_RERANKER_PROVIDER
    assert getattr(reranker, "_rag_core_provider_requested") == DEFAULT_RERANKER_PROVIDER
    assert getattr(reranker, "_rag_core_provider_effective") == DEFAULT_RERANKER_PROVIDER
    assert getattr(reranker, "_rag_core_fallback_reason") is None


@pytest.mark.parametrize(
    ("provider", "env_key", "expected_reason"),
    [
        ("cohere", "COHERE_API_KEY", "missing_cohere_api_key"),
        ("zeroentropy", "ZEROENTROPY_API_KEY", "missing_zeroentropy_api_key"),
    ],
)
def test_create_reranker_missing_key_falls_back_to_noop(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    env_key: str,
    expected_reason: str,
) -> None:
    monkeypatch.delenv(env_key, raising=False)
    monkeypatch.delenv("CO_API_KEY", raising=False)
    monkeypatch.delenv("RERANKER_STRICT_PROVIDER", raising=False)

    reranker = create_reranker(provider=provider)

    assert isinstance(reranker, NoOpReranker)
    assert getattr(reranker, "_rag_core_provider_requested") == provider
    assert getattr(reranker, "_rag_core_provider_effective") == DEFAULT_RERANKER_PROVIDER
    assert getattr(reranker, "_rag_core_fallback_reason") == expected_reason


def test_create_reranker_strict_mode_raises_instead_of_falling_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("CO_API_KEY", raising=False)
    monkeypatch.setenv("RERANKER_STRICT_PROVIDER", "true")

    with pytest.raises(ValueError, match="missing_cohere_api_key"):
        create_reranker(provider="cohere")


def test_create_reranker_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown reranker provider"):
        create_reranker(provider="made-up")


@pytest.mark.parametrize(
    ("builder", "provider_name", "model_name", "supports_page_selection", "module"),
    [
        (
            build_mistral_ocr_provider,
            MISTRAL_OCR_PROVIDER,
            DEFAULT_MISTRAL_OCR_MODEL,
            True,
            "rag_core.documents.ocr_commands.mistral",
        ),
        (
            build_gemini_ocr_provider,
            GEMINI_OCR_PROVIDER,
            DEFAULT_GEMINI_OCR_MODEL,
            False,
            "rag_core.documents.ocr_commands.gemini",
        ),
    ],
)
def test_ocr_builders_emit_python_module_command(
    builder: object,
    provider_name: str,
    model_name: str,
    supports_page_selection: bool,
    module: str,
) -> None:
    provider = builder(python_executable="/tmp/python")  # type: ignore[operator]

    assert provider.provider_name == provider_name
    assert provider.model_name == model_name
    assert provider.supports_page_selection is supports_page_selection
    assert provider._command == ["/tmp/python", "-m", module, "--model", model_name]


def test_env_helpers_fall_back_on_invalid_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_INT", "oops")
    monkeypatch.setenv("TEST_FLOAT", "oops")
    monkeypatch.setenv("TEST_BOOL", "maybe")
    monkeypatch.setenv("TEST_STRIPPED", "  value  ")

    assert get_env_int("TEST_INT", 7) == 7
    assert get_env_float("TEST_FLOAT", 1.5) == 1.5
    assert get_env_bool("TEST_BOOL", True) is True
    assert get_env_optional_bool("TEST_BOOL") is None
    assert get_env_stripped("TEST_STRIPPED") == "value"
