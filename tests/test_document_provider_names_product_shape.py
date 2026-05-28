from __future__ import annotations

from pathlib import Path

from rag_core.documents.contextualizer_provider_names import (
    ANTHROPIC_CONTEXTUALIZER_ID,
    CONTEXTUALIZER_DISABLED_ALIAS,
    CONTEXTUALIZER_PROVIDER_ORDER,
    DEFAULT_ANTHROPIC_CONTEXTUALIZER_MODEL,
    NOOP_CONTEXTUALIZER_ID,
)
from rag_core.documents.ocr_provider_names import (
    COMMAND_OCR_PROVIDER,
    DEFAULT_GEMINI_OCR_MODEL,
    DEFAULT_MISTRAL_OCR_MODEL,
    GEMINI_OCR_PROVIDER,
    MISTRAL_OCR_PROVIDER,
    OCR_PROVIDER_ORDER,
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


def test_ocr_provider_names_have_single_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/documents/ocr_provider_names.py",
            "src/rag_core/documents/ocr.py",
            "src/rag_core/documents/ocr_command_runtime.py",
            "src/rag_core/documents/ocr_commands/mistral.py",
            "src/rag_core/documents/ocr_commands/gemini.py",
            "src/rag_core/search/providers/provider_category_diagnostics.py",
            "src/rag_core/cli_doctor_output.py",
        )
    }

    assert MISTRAL_OCR_PROVIDER == "mistral"
    assert GEMINI_OCR_PROVIDER == "gemini"
    assert COMMAND_OCR_PROVIDER == "command"
    assert OCR_PROVIDER_ORDER == ("mistral", "gemini")
    assert DEFAULT_MISTRAL_OCR_MODEL == "mistral-ocr-latest"
    assert DEFAULT_GEMINI_OCR_MODEL == "gemini-2.5-flash"
    owner = sources["src/rag_core/documents/ocr_provider_names.py"]
    assert owner.count('MISTRAL_OCR_PROVIDER = "mistral"') == 1
    assert owner.count('GEMINI_OCR_PROVIDER = "gemini"') == 1
    assert owner.count('COMMAND_OCR_PROVIDER = "command"') == 1
    assert owner.count('DEFAULT_MISTRAL_OCR_MODEL = "mistral-ocr-latest"') == 1
    assert owner.count('DEFAULT_GEMINI_OCR_MODEL = "gemini-2.5-flash"') == 1
    combined_consumers = "\n".join(
        source
        for path, source in sources.items()
        if path != "src/rag_core/documents/ocr_provider_names.py"
    )
    assert "MISTRAL_OCR_PROVIDER" in combined_consumers
    assert "GEMINI_OCR_PROVIDER" in combined_consumers
    assert "COMMAND_OCR_PROVIDER" in combined_consumers
    assert "OCR_PROVIDER_ORDER" in combined_consumers
    assert "DEFAULT_MISTRAL_OCR_MODEL" in combined_consumers
    assert "DEFAULT_GEMINI_OCR_MODEL" in combined_consumers
    assert 'known=("mistral", "gemini")' not in combined_consumers
    assert '("ocr", ("mistral", "gemini"))' not in combined_consumers
    assert 'model_name: str = "mistral-ocr-latest"' not in combined_consumers
    assert 'model_name: str = "gemini-2.5-flash"' not in combined_consumers
    assert 'provider_name: str = "command"' not in combined_consumers
    assert 'default="mistral-ocr-latest"' not in combined_consumers
    assert 'default="gemini-2.5-flash"' not in combined_consumers





def test_contextualizer_provider_names_have_single_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/documents/contextualizer_provider_names.py",
            "src/rag_core/documents/contextualizer.py",
            "src/rag_core/documents/contextualizer_adapters.py",
            "src/rag_core/documents/contextualizer_anthropic_runtime.py",
            "src/rag_core/core_runtime.py",
            "src/rag_core/search/providers/provider_category_diagnostics.py",
            "src/rag_core/cli_doctor_output.py",
        )
    }

    assert NOOP_CONTEXTUALIZER_ID == "noop"
    assert CONTEXTUALIZER_DISABLED_ALIAS == "none"
    assert ANTHROPIC_CONTEXTUALIZER_ID == "anthropic"
    assert CONTEXTUALIZER_PROVIDER_ORDER == ("noop", "anthropic")
    assert DEFAULT_ANTHROPIC_CONTEXTUALIZER_MODEL == "claude-haiku-4-5-20251001"
    owner = sources["src/rag_core/documents/contextualizer_provider_names.py"]
    assert owner.count('NOOP_CONTEXTUALIZER_ID = "noop"') == 1
    assert owner.count('CONTEXTUALIZER_DISABLED_ALIAS = "none"') == 1
    assert owner.count('ANTHROPIC_CONTEXTUALIZER_ID = "anthropic"') == 1
    assert (
        owner.count(
            'DEFAULT_ANTHROPIC_CONTEXTUALIZER_MODEL = "claude-haiku-4-5-20251001"'
        )
        == 1
    )
    adapter = sources["src/rag_core/documents/contextualizer_adapters.py"]
    assert "contextualizer_id_prefix = ANTHROPIC_CONTEXTUALIZER_ID" in adapter
    assert "DEFAULT_ANTHROPIC_CONTEXTUALIZER_MODEL" in adapter
    assert "_DEFAULT_ANTHROPIC_MODEL" not in adapter
    assert (
        "_DEFAULT_ANTHROPIC_MODEL"
        not in sources["src/rag_core/documents/contextualizer_anthropic_runtime.py"]
    )
    assert 'model: str = "claude-haiku-4-5-20251001"' not in adapter
    runtime = sources["src/rag_core/core_runtime.py"]
    assert "NOOP_CONTEXTUALIZER_ID" in runtime
    assert 'normalized == "noop"' not in runtime
    diagnostics = sources[
        "src/rag_core/search/providers/provider_category_diagnostics.py"
    ]
    doctor_output = sources["src/rag_core/cli_doctor_output.py"]
    assert "CONTEXTUALIZER_PROVIDER_ORDER" in diagnostics
    assert "CONTEXTUALIZER_PROVIDER_ORDER" in doctor_output
    assert "CONTEXTUALIZER_DISABLED_ALIAS" in diagnostics
    assert "_CONTEXTUALIZER_DISABLED_ALIAS" not in diagnostics
    assert "_CONTEXTUALIZER_PROVIDER_ALIASES" not in diagnostics
    assert "_DISABLED_PROVIDER_ALIAS" not in diagnostics
    assert '"anthropic": "ANTHROPIC_API_KEY"' not in diagnostics
    assert '"anthropic": "anthropic"' not in diagnostics
    assert 'known=(NOOP_CONTEXTUALIZER_ID, "anthropic")' not in diagnostics
    assert '("contextualizer", ("noop", "anthropic"))' not in doctor_output
