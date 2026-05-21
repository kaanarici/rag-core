from __future__ import annotations

import pytest

from rag_core.cli import main
from rag_core.cli_output import search_hit_payload
from tests.support import make_search_result


def test_cli_top_level_help_excludes_removed_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    for removed in ("eval", "trace-summary"):
        assert f"  {removed} " not in f"\n{output}\n"
    assert "  serve " in f"\n{output}\n"


def test_search_hit_payload_matches_ragie_scored_chunk_fields() -> None:
    hit = make_search_result(
        document_id="doc-1",
        document_key="docs/guide.md",
        score=0.88,
    )
    payload = search_hit_payload(hit)
    for field in ("id", "text", "score", "document_id", "document_key"):
        assert field in payload


def test_beta_stable_tag_policy_documented() -> None:
    pyproject = open("pyproject.toml", encoding="utf-8").read()
    readme = open("README.md", encoding="utf-8").read()
    assert "Development Status :: 4 - Beta" in pyproject
    assert "rag-core" in readme
