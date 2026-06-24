from __future__ import annotations

import pytest

from rag_core.cli import _build_parser
from rag_core.core_models import Config


def test_cli_env_embedding_dimensions_rejects_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_CORE_EMBEDDING_DIMENSIONS", "0")
    parser = _build_parser()
    args = parser.parse_args(["doctor"])

    with pytest.raises(ValueError) as exc_info:
        Config.from_cli(args)

    assert str(exc_info.value) == (
        "EmbeddingConfig.dimensions must be a positive integer"
    )


def test_cli_env_embedding_dimensions_rejects_malformed_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_CORE_EMBEDDING_DIMENSIONS", "foo")
    parser = _build_parser()
    args = parser.parse_args(["doctor"])

    with pytest.raises(ValueError) as exc_info:
        Config.from_cli(args)

    assert str(exc_info.value) == "RAG_CORE_EMBEDDING_DIMENSIONS must be an integer"
