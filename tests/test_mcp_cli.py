from __future__ import annotations

import argparse
import asyncio
import builtins
from typing import NoReturn

import pytest

from rag_core.cli_mcp import run_mcp_command
from rag_core.cli_parser import _build_parser
from rag_core.config import DEFAULT_VECTOR_STORE_PROVIDER
from rag_core.core_models import RAGCoreConfig


def test_mcp_command_parser_reuses_config_and_scope_flags() -> None:
    parser = _build_parser()

    args = parser.parse_args(
        [
            "mcp",
            "--namespace",
            "acme",
            "--corpus-id",
            "help",
            "--qdrant-location",
            ":memory:",
            "--embedding-provider",
            "demo",
            "--embedding-dimensions",
            "64",
            "--limit-cap",
            "7",
            "--rerank",
        ]
    )

    assert args.command == "mcp"
    assert args.namespace == "acme"
    assert args.corpus_id == ["help"]
    assert args.qdrant_location == ":memory:"
    assert args.vector_store == DEFAULT_VECTOR_STORE_PROVIDER
    assert args.embedding_provider == "demo"
    assert args.embedding_dimensions == 64
    assert args.limit_cap == 7
    assert args.rerank is True


def test_mcp_command_reports_missing_extra_before_runtime_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def patched_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "mcp.server.stdio":
            raise ImportError("No module named 'mcp'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", patched_import)
    args = argparse.Namespace(
        corpus_id=["help"],
        namespace="acme",
        limit_cap=5,
        rerank=False,
    )

    with pytest.raises(SystemExit, match="rag-core mcp requires the mcp extra"):
        asyncio.run(run_mcp_command(args, core_factory=_fail_core_factory))


def _fail_core_factory(_config: RAGCoreConfig) -> NoReturn:
    raise AssertionError("core must not be constructed when mcp extra is missing")
