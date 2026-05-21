from __future__ import annotations

from pathlib import Path

import pytest

import rag_core
from rag_core.cli import main


def test_demo_factory_is_not_root_public_api() -> None:
    assert "build_demo_core" not in rag_core.__all__
    assert not hasattr(rag_core, "build_demo_core")


def test_rag_core_facade_modules_live_under_facade_package() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "rag_core"

    assert not list(root.glob("core_facade_*.py"))
    assert sorted(path.name for path in (root / "facade").glob("*.py")) == [
        "__init__.py",
        "ingest.py",
        "ingest_batches.py",
        "ingest_sources.py",
        "manifest.py",
        "prepare.py",
        "retrieval.py",
    ]


@pytest.mark.parametrize(
    "command",
    ("doctor", "ingest", "ingest-url", "search", "eval"),
)
def test_cli_help_uses_vector_store_language(
    command: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main([command, "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "configured Qdrant collection" not in output
