from __future__ import annotations

import argparse
from typing import Any

import pytest

from rag_core.cli.parsers.config import add_config_flags
from rag_core.core_models import Config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_config_flags(parser)
    return parser


def test_qdrant_api_key_flag_emits_warning(capsys) -> None:
    parser = _parser()

    parser.parse_args(["--qdrant-api-key", "secret"])

    err = capsys.readouterr().err
    assert "--qdrant-api-key exposes credentials" in err
    assert "RAG_CORE_QDRANT_API_KEY env var" in err


@pytest.mark.parametrize(
    "url",
    [
        "https://user:pass@example.qdrant.io",
        "https://example.qdrant.io?api_key=secret",
        "https://example.qdrant.io?token=secret",
    ],
)
def test_qdrant_url_flag_warns_when_url_contains_credentials(
    url: str,
    capsys: Any,
) -> None:
    parser = _parser()

    parser.parse_args(["--qdrant-url", url])

    err = capsys.readouterr().err
    assert "--qdrant-url contains credentials" in err
    assert "RAG_CORE_QDRANT_URL env var" in err


@pytest.mark.parametrize(
    ("env_name", "env_value", "message"),
    [
        (
            "RAG_CORE_EMBEDDING_BATCH_SIZE",
            "not-an-int",
            "RAG_CORE_EMBEDDING_BATCH_SIZE must be an integer",
        ),
        (
            "RAG_CORE_EMBEDDING_DIMENSIONS",
            "not-an-int",
            "RAG_CORE_EMBEDDING_DIMENSIONS must be an integer",
        ),
        (
            "RAG_CORE_QDRANT_DIMENSION_AWARE_COLLECTION",
            "maybe",
            "RAG_CORE_QDRANT_DIMENSION_AWARE_COLLECTION must be a boolean",
        ),
    ],
)
def test_config_env_parsing_rejects_invalid_values(
    env_name: str,
    env_value: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(env_name, env_value)

    args = _parser().parse_args([])

    with pytest.raises(ValueError, match=message):
        Config.from_cli(args)


@pytest.mark.parametrize(
    ("env_name", "env_value", "args", "assert_config"),
    [
        (
            "RAG_CORE_EMBEDDING_BATCH_SIZE",
            "not-an-int",
            ["--embedding-batch-size", "2"],
            lambda config: config.embedding.batch_size == 2,
        ),
        (
            "RAG_CORE_EMBEDDING_DIMENSIONS",
            "not-an-int",
            ["--embedding-dimensions", "512"],
            lambda config: config.embedding.dimensions == 512,
        ),
        (
            "RAG_CORE_QDRANT_DIMENSION_AWARE_COLLECTION",
            "maybe",
            ["--no-dimension-aware-collection"],
            lambda config: config.qdrant.dimension_aware_collection is False,
        ),
    ],
)
def test_explicit_cli_flags_override_invalid_config_env(
    env_name: str,
    env_value: str,
    args: list[str],
    assert_config: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(env_name, env_value)

    config = Config.from_cli(_parser().parse_args(args))

    assert assert_config(config)
