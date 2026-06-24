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
from rag_core import Engine, Config
from rag_core.search.providers.diagnostic_support import (
    PROVIDER_DIAGNOSTIC_MATURITIES,
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


def _provider_doc_extras() -> tuple[str, ...]:
    converter_only = {"pdf"}
    return tuple(extra for extra in _optional_extras() if extra not in converter_only)


def _phase1_normalize_api_names(value: str) -> str:
    return value.replace("RAGCoreConfig", "Config").replace("RAGCore", "Engine")


def _custom_provider_rows() -> dict[str, dict[str, str]]:
    docs = _read("docs-site/content/docs/providers.mdx")
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
            "runtime_selection": _phase1_normalize_api_names(runtime_selection),
        }
    return rows


def _vector_store_rows() -> dict[str, dict[str, str]]:
    docs = _read("docs-site/content/docs/providers.mdx")
    _, table = docs.split("| Provider | Maturity | Entrypoint |", 1)
    rows: dict[str, dict[str, str]] = {}
    for raw in table.splitlines()[2:]:
        if not raw.startswith("| "):
            break
        cells = [cell.strip() for cell in raw.strip("|").split("|")]
        provider, maturity, entrypoint = cells
        rows[provider] = {
            "maturity": maturity,
            "entrypoint": _phase1_normalize_api_names(entrypoint),
        }
    return rows


DOC_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "README.md",
        (
            "uv pip install -e .",
            "./scripts/dx_smoke.sh",
            "https://kaanarici.github.io/rag-core/docs/api-python",
            "uv run python -m examples.minimal_app",
            "examples/retrieval_eval.py",
            "examples/configured_retrieval.py",
            "https://kaanarici.github.io/rag-core/docs/stability",
        ),
    ),
    (
        "docs-site/content/docs/providers.mdx",
        (
            "QdrantConfig",
            "Qdrant as the built-in store",
            "the in-memory vector store",
            "EMBEDDING_PROVIDERS",
            "VECTOR_STORES",
            "RERANKER_PROVIDERS",
            "CONTEXTUALIZER_PROVIDERS",
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
    docs = _read("docs-site/content/docs/providers.mdx")
    for extra in _provider_doc_extras():
        assert f"`{extra}`" in docs


def test_provider_diagnostics_list_shared_maturities_without_legacy_values() -> None:
    for maturity in PROVIDER_DIAGNOSTIC_MATURITIES:
        assert not maturity.startswith("first_party_")
    assert "disabled" in PROVIDER_DIAGNOSTIC_MATURITIES
    assert "optional" in PROVIDER_DIAGNOSTIC_MATURITIES
    assert "utility" in PROVIDER_DIAGNOSTIC_MATURITIES


def test_vector_store_maturity_table_matches_typed_provider_specs() -> None:
    rows = _vector_store_rows()
    expected = {spec.docs_label: spec for spec in BUILTIN_VECTOR_STORE_PROVIDER_SPECS}
    expected_docs_maturity = {
        "Qdrant": "stable",
        "pgvector": "beta",
        "TurboPuffer": "beta",
        "In-memory": "beta",
    }

    assert set(rows) == set(expected)
    for label in expected:
        assert rows[label]["maturity"] == expected_docs_maturity[label]


def test_turbopuffer_docs_describe_real_adapter_behavior() -> None:
    docs = _read("docs-site/content/docs/providers.mdx")
    turbopuffer_section = docs.split("### TurboPuffer", 1)[1].split(
        "### pgvector", 1
    )[0]
    normalized = " ".join(turbopuffer_section.split())

    # The section honestly scopes the beta adapter: an optional extra requiring a
    # key, BM25 as the lexical channel, client-side hybrid RRF, and unsupported
    # query-plan stages failing closed rather than silently degrading.
    assert "beta adapter" in normalized
    assert "TURBOPUFFER_API_KEY" in normalized
    assert "BM25" in normalized
    assert "hybrid RRF" in normalized
    assert "fail closed" in normalized

    # Guardrail: do not invent fixture/proof artifacts that do not exist.
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
        "Chunk contextualizers": (
            "CONTEXTUALIZER_PROVIDERS",
            documents.CONTEXTUALIZER_PROVIDERS,
        ),
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
        # rag_core.search.types was a deleted shim. Owners live in
        # rag_core.search.provider_protocols. Assert the symbol resolves.
        assert getattr(provider_protocols, name) is not None

    assert rows["Converters"]["registry_or_builtins"] == (
        "Dedicated loader: `get_converter()`, "
        "`rag_core.documents.converters.convert_file()`"
    )
    assert converters.__all__ == (
        "BaseConverter",
        "ConversionResult",
        "QualityVerdict",
        "convert_file",
        "get_converter",
    )
    for contextualizer_name in (
        "NoOpContextualizer",
        "AnthropicChunkContextualizer",
        "CachingContextualizer",
    ):
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
    rag_core_params = set(inspect.signature(Engine).parameters)
    config_fields = {field.name for field in fields(Config)}

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
        assert f"Engine({parameter}=...)" in rows[category]["runtime_selection"]
        assert parameter in rag_core_params

    documented_config_roots = {
        "Dense embeddings": "embedding",
        "Rerankers": "reranker",
        "Chunk contextualizers": "contextualizer",
        "Search sidecars": "ingest",
        "Embedding cache": "ingest",
    }
    for category, root_field in documented_config_roots.items():
        assert "Config" in rows[category]["runtime_selection"]
        assert root_field in config_fields

    vector_store_runtime = rows["Vector stores"]["runtime_selection"]
    assert (
        "Qdrant/pgvector/TurboPuffer via `Config.vector_store`"
        in vector_store_runtime
    )
    assert "CLI" in vector_store_runtime
    assert "`Engine(vector_store=...)`" in vector_store_runtime
    assert "vector_store" in config_fields

    assert "no `Config` field" in rows["OCR"]["runtime_selection"]
    assert rows["Chunk context cache"]["runtime_selection"] == (
        "`Engine(chunk_context_cache=...)` only"
    )
    assert "Config.contextualizer" in rows["Chunk contextualizers"]["runtime_selection"]
    assert rows["Event sinks"]["runtime_selection"] == "`Engine(event_sink=...)` only"
    assert "prepare paths" in rows["Chunking"]["runtime_selection"]
