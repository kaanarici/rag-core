from __future__ import annotations

import pytest

from rag_core.documents.chunking.code import CodeChunker
from rag_core.documents.chunking.semantic import SemanticChunker


_CHUNKING_ENV_NAMES = (
    "CHUNKING_SKIP_UNSUPPORTED_CODE",
    "CHUNKING_ENABLE_MAGIKA_LANGUAGE_DETECTION",
    "CHUNKING_ENABLE_LOCAL_SEMANTIC",
)


def _clear_chunking_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _CHUNKING_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_invalid_magika_detection_env_preserves_enabled_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_chunking_env(monkeypatch)
    monkeypatch.setenv("CHUNKING_ENABLE_MAGIKA_LANGUAGE_DETECTION", "maybe")

    chunker = CodeChunker()

    assert chunker._enable_magika_detection is True


def test_false_magika_detection_env_disables_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_chunking_env(monkeypatch)
    monkeypatch.setenv("CHUNKING_ENABLE_MAGIKA_LANGUAGE_DETECTION", "false")

    chunker = CodeChunker()

    assert chunker._enable_magika_detection is False


def test_true_skip_unsupported_code_env_enables_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_chunking_env(monkeypatch)
    monkeypatch.setenv("CHUNKING_SKIP_UNSUPPORTED_CODE", "yes")

    chunker = CodeChunker()

    assert chunker._skip_unsupported_language is True


def test_invalid_skip_unsupported_code_env_preserves_disabled_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_chunking_env(monkeypatch)
    monkeypatch.setenv("CHUNKING_SKIP_UNSUPPORTED_CODE", "maybe")

    chunker = CodeChunker()

    assert chunker._skip_unsupported_language is False


def test_true_local_semantic_env_enables_local_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_chunking_env(monkeypatch)
    monkeypatch.setenv("CHUNKING_ENABLE_LOCAL_SEMANTIC", "1")

    chunker = SemanticChunker()

    assert chunker._enable_local_model is True


def test_invalid_local_semantic_env_preserves_disabled_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_chunking_env(monkeypatch)
    monkeypatch.setenv("CHUNKING_ENABLE_LOCAL_SEMANTIC", "maybe")

    chunker = SemanticChunker()

    assert chunker._enable_local_model is False
