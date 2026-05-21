from __future__ import annotations

import re

import pytest

from rag_core.cli_parser import _build_parser


def _command_help(argv: list[str], capsys: pytest.CaptureFixture[str]) -> str:
    parser = _build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args([*argv, "--help"])
    assert exc_info.value.code == 0
    return capsys.readouterr().out


def _flat_help(output: str) -> str:
    unhyphenated = re.sub(r"-\n\s*", "-", output)
    return re.sub(r"\s+", " ", unhyphenated)


def test_query_help_describes_profiles_and_presets(
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = _flat_help(_command_help(["search"], capsys))

    assert "balanced=general-purpose hybrid retrieval" in output
    assert "fast=low-latency semantic retrieval" in output
    assert "hybrid_rrf=dense plus sparse retrieval fused with reciprocal" in output
    assert "hybrid_with_mmr=hybrid reciprocal-rank fusion followed by MMR" in output


def test_search_help_lists_extended_profiles(
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = _flat_help(_command_help(["search"], capsys))

    assert "coverage=hybrid retrieval with score-distribution fusion" in output
    assert "diverse=hybrid retrieval with diversity reranking" in output
