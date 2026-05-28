from __future__ import annotations

import inspect
import tomllib
from dataclasses import fields
from pathlib import Path

import pytest

import rag_core.documents as documents
import rag_core.documents.chunking as chunking
import rag_core.documents.converters as converters
import rag_core.events as events
import rag_core.search.provider_protocols as provider_protocols
import rag_core.search.providers as providers
import rag_core.search.types as search_types
from rag_core import RAGCore, RAGCoreConfig
from rag_core.search.providers.diagnostic_support import (
    PROVIDER_DIAGNOSTIC_SUPPORT_LEVELS,
)
from rag_core.search.providers.vector_store_capabilities import (
    BUILTIN_VECTOR_STORE_PROVIDER_SPECS,
)
from rag_core.search.providers.registry import ProviderRegistry

pytestmark = [pytest.mark.meta]


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _optional_extras() -> tuple[str, ...]:
    pyproject = tomllib.loads(_read("pyproject.toml"))
    return tuple(pyproject["project"]["optional-dependencies"])


def _custom_provider_rows() -> dict[str, dict[str, str]]:
    docs = _read("docs/providers.md")
    _, table = docs.split(
        "| Category | Protocol | Registry or built-ins | Runtime selection |",
        1,
    )
    rows: dict[str, dict[str, str]] = {}
    for raw in table.splitlines()[2:]:
        if not raw.startswith("| "):
            break
        cells = [cell.strip() for cell in raw.strip("|").split("|")]
        category, protocol, registry_or_builtins, runtime_selection = cells
        rows[category] = {
            "protocol": protocol.strip("`"),
            "registry_or_builtins": registry_or_builtins,
            "runtime_selection": runtime_selection,
        }
    return rows


def _vector_store_rows() -> dict[str, dict[str, str]]:
    docs = _read("docs/providers.md")
    _, table = docs.split("| Provider | Maturity | Entrypoint |", 1)
    rows: dict[str, dict[str, str]] = {}
    for raw in table.splitlines()[2:]:
        if not raw.startswith("| "):
            break
        cells = [cell.strip() for cell in raw.strip("|").split("|")]
        provider, maturity, entrypoint = cells
        rows[provider] = {
            "maturity": maturity,
            "entrypoint": entrypoint,
        }
    return rows


DOC_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "README.md",
        (
            "uv pip install -e .",
            "./scripts/dx_smoke.sh",
            "docs/embed.md",
            "uv run python -m examples.minimal_app",
            "examples/retrieval_eval.py",
            "examples/configured_retrieval.py",
            "docs/stability.md",
        ),
    ),
    (
        "docs/embed.md",
        ("from rag_core.demo import build_demo_core",),
    ),
    (
        "docs/providers.md",
        (
            "QdrantConfig",
            "default wheel** ships **Qdrant**",
            "the in-memory vector store",
            "EMBEDDING_PROVIDERS",
            "VECTOR_STORES",
            "RERANKER_PROVIDERS",
        ),
    ),
)


@pytest.mark.parametrize(("path", "terms"), DOC_TERMS)
def test_provider_docs_match_current_install_story(
    path: str,
    terms: tuple[str, ...],
) -> None:
    body = _read(path)
    for term in terms:
        assert term in body


def test_provider_docs_name_optional_provider_extras() -> None:
    docs = _read("docs/providers.md")
    for extra in _optional_extras():
        assert f"`{extra}`" in docs


def test_provider_docs_list_shared_diagnostic_support_levels() -> None:
    docs = _read("docs/providers.md")

    for support_level in PROVIDER_DIAGNOSTIC_SUPPORT_LEVELS:
        assert f"| `{support_level}` |" in docs


def test_vector_store_maturity_table_matches_typed_provider_specs() -> None:
    rows = _vector_store_rows()
    expected = {spec.docs_label: spec for spec in BUILTIN_VECTOR_STORE_PROVIDER_SPECS}

    assert set(rows) == set(expected)
    for label, spec in expected.items():
        assert rows[label] == {
            "maturity": spec.docs_maturity,
            "entrypoint": spec.docs_entrypoint,
        }


def test_turbopuffer_docs_name_actual_adapter_proof() -> None:
    docs = _read("docs/providers.md")
    turbopuffer_section = docs.split("### TurboPuffer (optional)", 1)[1].split(
        "### Migration", 1
    )[0]

    for path in (
        "tests/test_turbopuffer_store.py",
        "tests/test_turbopuffer_query_plan_guard.py",
        "tests/test_turbopuffer_result_shape_validation.py",
        "tests/test_vector_store_contract.py",
    ):
        assert path in turbopuffer_section
    assert "`provider_contract`" not in turbopuffer_section
    assert "JSON fixtures" not in turbopuffer_section


def test_custom_provider_table_matches_extension_selection_surface() -> None:
    rows = _custom_provider_rows()

    expected_registries = {
        "Dense embeddings": ("EMBEDDING_PROVIDERS", providers.EMBEDDING_PROVIDERS),
        "Sparse embeddings": ("SPARSE_EMBEDDERS", providers.SPARSE_EMBEDDERS),
        "Rerankers": ("RERANKER_PROVIDERS", providers.RERANKER_PROVIDERS),
        "Vector stores": ("VECTOR_STORES", providers.VECTOR_STORES),
        "OCR": ("OCR_PROVIDERS", documents.OCR_PROVIDERS),
        "Search sidecars": ("SEARCH_SIDECARS", providers.SEARCH_SIDECARS),
        "Embedding cache": ("EMBEDDING_CACHES", providers.EMBEDDING_CACHES),
        "Chunk context cache": (
            "CHUNK_CONTEXT_CACHES",
            providers.CHUNK_CONTEXT_CACHES,
        ),
        "Chunking": ("CHUNKING_STRATEGIES", chunking.CHUNKING_STRATEGIES),
    }
    assert set(expected_registries).issubset(rows)
    for category, (documented_name, registry) in expected_registries.items():
        assert isinstance(registry, ProviderRegistry)
        assert rows[category]["registry_or_builtins"] == f"`{documented_name}`"

    expected_protocols = {
        "Dense embeddings": "EmbeddingProvider",
        "Sparse embeddings": "SparseEmbedder",
        "Rerankers": "RerankerProvider",
        "Vector stores": "VectorStore",
        "Search sidecars": "SearchSidecar",
    }
    for category, name in expected_protocols.items():
        assert rows[category]["protocol"] == (
            f"rag_core.search.provider_protocols.{name}"
        )
        assert getattr(provider_protocols, name) is getattr(search_types, name)

    assert rows["Converters"]["registry_or_builtins"] == (
        "Dedicated loader: `get_converter()`, `convert_file()`"
    )
    assert converters.__all__ == (
        "BaseConverter",
        "ConversionResult",
        "QualityVerdict",
        "convert_file",
        "get_converter",
    )
    contextualizer_builtins = rows["Chunk contextualizers"]["registry_or_builtins"]
    assert contextualizer_builtins.startswith("Built-ins: ")
    for contextualizer_name in (
        "NoOpContextualizer",
        "AnthropicChunkContextualizer",
        "CachingContextualizer",
    ):
        assert f"`{contextualizer_name}`" in contextualizer_builtins
        assert contextualizer_name in documents.__all__
    event_sink_builtins = rows["Event sinks"]["registry_or_builtins"]
    assert event_sink_builtins.startswith("Built-ins: ")
    for sink_name in (
        "NoOpSink",
        "LoggingSink",
        "JsonlSink",
        "EventBuffer",
        "MultiSink",
        "OpenTelemetrySink",
    ):
        assert f"`{sink_name}`" in event_sink_builtins
        assert sink_name in events.__all__


def test_custom_provider_table_matches_runtime_injection_surface() -> None:
    rows = _custom_provider_rows()
    rag_core_params = set(inspect.signature(RAGCore).parameters)
    config_fields = {field.name for field in fields(RAGCoreConfig)}

    documented_injections = {
        "Dense embeddings": "embedding_provider",
        "Sparse embeddings": "sparse_embedder",
        "Rerankers": "reranker",
        "Vector stores": "vector_store",
        "OCR": "ocr_provider",
        "Search sidecars": "search_sidecar",
        "Embedding cache": "embedding_cache",
        "Chunk context cache": "chunk_context_cache",
        "Chunk contextualizers": "chunk_contextualizer",
        "Event sinks": "event_sink",
    }
    for category, parameter in documented_injections.items():
        assert f"RAGCore({parameter}=...)" in rows[category]["runtime_selection"]
        assert parameter in rag_core_params

    documented_config_roots = {
        "Dense embeddings": "embedding",
        "Rerankers": "reranker",
        "Search sidecars": "ingest",
        "Embedding cache": "ingest",
    }
    for category, root_field in documented_config_roots.items():
        assert "RAGCoreConfig" in rows[category]["runtime_selection"]
        assert root_field in config_fields

    vector_store_runtime = rows["Vector stores"]["runtime_selection"]
    assert "Qdrant/TurboPuffer via `RAGCoreConfig.vector_store`" in vector_store_runtime
    assert "CLI" in vector_store_runtime
    assert "`RAGCore(vector_store=...)`" in vector_store_runtime
    assert "vector_store" in config_fields

    assert "no `RAGCoreConfig` field" in rows["OCR"]["runtime_selection"]
    assert rows["Chunk context cache"]["runtime_selection"] == (
        "`RAGCore(chunk_context_cache=...)` only"
    )
    assert rows["Chunk contextualizers"]["runtime_selection"] == (
        "`RAGCore(chunk_contextualizer=...)` only"
    )
    assert rows["Event sinks"]["runtime_selection"] == "`RAGCore(event_sink=...)` only"
    assert "prepare paths" in rows["Chunking"]["runtime_selection"]
