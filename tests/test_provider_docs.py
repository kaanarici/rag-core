from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

pytestmark = [pytest.mark.meta]


def test_provider_docs_match_current_install_story() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    provider_docs = Path("docs/providers.md").read_text(encoding="utf-8")

    assert 'uv add "rag-core @ git+https://github.com/kaanarici/rag-core.git"' in readme
    assert "QdrantConfig" in provider_docs
    assert "default wheel** ships **Qdrant**" in provider_docs


def test_readme_documents_smoke_configured_and_embed_entrypoints() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    embed = Path("docs/embed.md").read_text(encoding="utf-8")

    assert "./scripts/dx_smoke.sh" in readme
    assert "docs/embed.md" in readme
    assert "from rag_core.demo import build_demo_core" in embed
    assert "uv run python -m examples.minimal_app" in readme
    assert "examples/retrieval_eval.py" in readme
    assert "examples/configured_retrieval.py" in readme
    assert "docs/stability.md" in readme


def test_provider_docs_name_extension_point_registries() -> None:
    docs = Path("docs/providers.md").read_text(encoding="utf-8")
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    for term in ("EMBEDDING_PROVIDERS", "VECTOR_STORES", "RERANKER_PROVIDERS"):
        assert term in docs
    for extra in pyproject["project"]["optional-dependencies"]:
        assert f"`{extra}`" in docs
