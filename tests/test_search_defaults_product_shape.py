from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from rag_core.retrieval_defaults import (
    DEFAULT_CONTEXT_LIMIT,
    DEFAULT_LOCAL_SEARCH_LIMIT,
    DEFAULT_RERANK,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_USE_LEXICAL_SEARCH,
)
from rag_core.local_search_models import (
    DEFAULT_LOCAL_SEARCH_COLLECTION,
    DEFAULT_LOCAL_SEARCH_NAMESPACE,
)
from rag_core.retrieval_channels import (
    DENSE_RETRIEVAL_CHANNEL,
    RETRIEVAL_CHANNELS,
    SPARSE_RETRIEVAL_CHANNEL,
)
from rag_core.search import DenseChannel, Prefetch, QueryPlan, default_query_plan
from rag_core.search.pipeline_runner import SearchRequest
from rag_core.search.request_models import SearchQuery, SearchSidecarQuery
from rag_core.search.stored_payload_fields import (
    SEARCH_RESULT_FILTER_FIELDS,
    SEARCH_RESULT_LEGACY_THUMBNAIL_FIELD,
    SEARCH_RESULT_STORED_METADATA_FIELDS,
)
from rag_core.search.types import SEARCH_RESULT_TYPE_TEXT, SparseVector
from tests.support import make_search_result


def test_lexical_search_default_has_single_retrieval_owner() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = {
        path: (root / path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/retrieval_defaults.py",
            "src/rag_core/core_retrieval.py",
            "src/rag_core/facade/retrieval.py",
            "src/rag_core/runtime/requests.py",
            "src/rag_core/contracts/tool_contract_schemas.py",
            "src/rag_core/search/pipeline_runner.py",
            "src/rag_core/search/pipeline/types.py",
        )
    }

    assert DEFAULT_USE_LEXICAL_SEARCH is True
    assert (
        sources["src/rag_core/retrieval_defaults.py"].count(
            "DEFAULT_USE_LEXICAL_SEARCH = True"
        )
        == 1
    )
    for path, source in sources.items():
        if path == "src/rag_core/retrieval_defaults.py":
            continue
        assert "DEFAULT_USE_LEXICAL_SEARCH" in source
        assert "use_lexical_search: bool = True" not in source
        assert "default=True" not in source
    assert (
        "SEARCH_USER_DOCUMENTS_DEFAULT_USE_LEXICAL_SEARCH: Final[bool] = "
        "DEFAULT_USE_LEXICAL_SEARCH"
    ) in sources["src/rag_core/contracts/tool_contract_schemas.py"]



def test_rerank_default_has_single_retrieval_owner() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = {
        path: (root / path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/retrieval_defaults.py",
            "src/rag_core/core_retrieval.py",
            "src/rag_core/facade/retrieval.py",
            "src/rag_core/runtime/requests.py",
            "src/rag_core/contracts/tool_contract_schemas.py",
            "src/rag_core/search/pipeline_runner.py",
            "src/rag_core/search/pipeline/types.py",
            "src/rag_core/evals/runner.py",
            "examples/corpus_lifecycle.py",
        )
    }

    assert DEFAULT_RERANK is False
    assert (
        sources["src/rag_core/retrieval_defaults.py"].count("DEFAULT_RERANK = False")
        == 1
    )
    for path, source in sources.items():
        if path == "src/rag_core/retrieval_defaults.py":
            continue
        assert "DEFAULT_RERANK" in source
        assert "    rerank: bool = False" not in source
        assert '"rerank", default=False' not in source
    assert (
        "SEARCH_USER_DOCUMENTS_DEFAULT_RERANK: Final[bool] = DEFAULT_RERANK"
        in sources["src/rag_core/contracts/tool_contract_schemas.py"]
    )



def test_local_search_runner_does_not_introduce_a_parallel_search_plan() -> None:
    root = Path(__file__).resolve().parents[1]
    source = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/local_search_planning.py",
            "src/rag_core/local_search_runner.py",
        )
    )

    assert "LocalSearchRunSpec" in source
    assert "build_local_search_run_spec" in source
    assert "LocalSearchPlan" not in source
    assert "build_local_search_plan" not in source



def test_local_search_hit_payload_names_local_projection() -> None:
    root = Path(__file__).resolve().parents[1]
    local_runner = (root / "src" / "rag_core" / "local_search_runner.py").read_text(
        encoding="utf-8"
    )
    local_ingest = (root / "src" / "rag_core" / "local_ingest.py").read_text(
        encoding="utf-8"
    )
    cli_output = (root / "src" / "rag_core" / "cli_output.py").read_text(
        encoding="utf-8"
    )

    assert "def search_hit_payload(" in cli_output
    assert "def local_search_hit_payload(" in local_runner
    assert "local_search_hit_payload(hit)" in local_runner
    assert '"local_search_hit_payload"' in local_runner
    assert "local_search_hit_payload" in local_ingest
    assert "def search_hit_payload(" not in local_runner
    assert '"search_hit_payload"' not in local_runner
    assert " search_hit_payload" not in local_ingest



def test_public_entrypoint_defaults_are_named_once() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/cli_local_search_parser.py",
            "src/rag_core/cli_search_parser.py",
            "src/rag_core/core_retrieval.py",
            "src/rag_core/facade/retrieval.py",
            "src/rag_core/local_search_models.py",
            "src/rag_core/runtime/requests.py",
            "src/rag_core/search/context_pack.py",
        )
    )

    assert DEFAULT_SEARCH_LIMIT == 10
    assert DEFAULT_LOCAL_SEARCH_LIMIT == 5
    assert DEFAULT_CONTEXT_LIMIT == 8
    assert "DEFAULT_SEARCH_LIMIT" in sources
    assert "DEFAULT_LOCAL_SEARCH_LIMIT" in sources
    assert "DEFAULT_CONTEXT_LIMIT" in sources
    assert "limit: int = 10" not in sources
    assert "limit: int = 8" not in sources
    assert "limit: int = 5" not in sources
    assert '--limit", type=int, default=10' not in sources
    assert '--limit", type=int, default=8' not in sources
    assert '--limit", type=int, default=5' not in sources



def test_local_search_string_defaults_have_single_owner() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = {
        path: (root / path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/cli_local_search.py",
            "src/rag_core/cli_local_search_parser.py",
            "src/rag_core/local_search_models.py",
            "src/rag_core/local_search_runner.py",
        )
    }

    assert DEFAULT_LOCAL_SEARCH_COLLECTION == "local_search"
    assert DEFAULT_LOCAL_SEARCH_NAMESPACE == "local"
    assert (
        'DEFAULT_LOCAL_SEARCH_COLLECTION = "local_search"'
        in sources["src/rag_core/local_search_models.py"]
    )
    assert (
        'DEFAULT_LOCAL_SEARCH_NAMESPACE = "local"'
        in sources["src/rag_core/local_search_models.py"]
    )
    for path, source in sources.items():
        if path == "src/rag_core/local_search_models.py":
            continue
        assert 'collection="local_search"' not in source
        assert 'default="local"' not in source
        assert 'namespace: str = "local"' not in source



def test_internal_search_pipeline_defaults_use_named_search_limit() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/search/pipeline_runner.py",
            "src/rag_core/search/pipeline/types.py",
            "src/rag_core/search/query_plan.py",
            "src/rag_core/search/query_plan_presets.py",
            "src/rag_core/search/request_models.py",
            "src/rag_core/search/provider_protocols.py",
            "src/rag_core/search/providers/reranker.py",
            "src/rag_core/search/providers/cohere.py",
            "src/rag_core/search/providers/voyage.py",
            "src/rag_core/search/providers/zeroentropy.py",
            "src/rag_core/evals/runner.py",
            "tests/support/turbopuffer_fake.py",
        )
    )

    assert SearchRequest(query="q", corpus_ids=["c"], namespace="n").limit == (
        DEFAULT_SEARCH_LIMIT
    )
    assert (
        SearchQuery(
            dense_vector=[1.0],
            sparse_vector=SparseVector(indices=[1], values=[1.0]),
            namespace="n",
            corpus_ids=["c"],
        ).limit
        == DEFAULT_SEARCH_LIMIT
    )
    assert (
        SearchSidecarQuery(query="q", namespace="n", corpus_ids=["c"]).limit
        == DEFAULT_SEARCH_LIMIT
    )
    assert (
        QueryPlan(prefetches=(Prefetch(channel=DenseChannel(), limit=10),)).final_limit
        == DEFAULT_SEARCH_LIMIT
    )
    assert default_query_plan().final_limit == DEFAULT_SEARCH_LIMIT
    assert "limit: int = DEFAULT_SEARCH_LIMIT" in sources
    assert "result_limit: int = DEFAULT_SEARCH_LIMIT" in sources
    assert "top_k: int = DEFAULT_SEARCH_LIMIT" in sources
    assert "DEFAULT_SEARCH_LIMIT" in sources
    assert "limit: int = 20" not in sources
    assert "result_limit: int = 20" not in sources
    assert "top_k: int = 10" not in sources
    assert "if k_values else 10" not in sources
    assert 'or kwargs.get("limit") or 10' not in sources



def test_retrieval_channel_labels_have_single_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/retrieval_channels.py",
            "src/rag_core/events/document_events.py",
            "src/rag_core/events/embedding_trace_summary.py",
            "src/rag_core/cli_doctor_output.py",
            "src/rag_core/search/indexer_embeddings.py",
            "src/rag_core/search/pipeline/stages/hybrid_retrieve.py",
            "src/rag_core/search/query_plan_presets.py",
            "src/rag_core/search/query_plan_trace.py",
        )
    }

    assert DENSE_RETRIEVAL_CHANNEL == "dense"
    assert SPARSE_RETRIEVAL_CHANNEL == "sparse"
    assert RETRIEVAL_CHANNELS == ("dense", "sparse")
    owner = sources["src/rag_core/retrieval_channels.py"]
    assert (
        owner.count('DENSE_RETRIEVAL_CHANNEL: Final[RetrievalChannel] = "dense"') == 1
    )
    assert (
        owner.count('SPARSE_RETRIEVAL_CHANNEL: Final[RetrievalChannel] = "sparse"') == 1
    )
    for path, source in sources.items():
        if path == "src/rag_core/retrieval_channels.py":
            continue
        assert (
            "DENSE_RETRIEVAL_CHANNEL" in source or "SPARSE_RETRIEVAL_CHANNEL" in source
        )
        assert 'role="dense"' not in source
        assert 'role="sparse"' not in source
        assert 'event.role == "dense"' not in source
        assert 'event.role == "sparse"' not in source
        assert 'channels=("dense"' not in source
        assert 'field = channel.vector_field or "dense"' not in source



def test_search_result_type_text_label_has_single_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/search/vector_models.py",
            "src/rag_core/search/types.py",
            "src/rag_core/search/context_pack_helpers.py",
            "src/rag_core/search/stored_payload.py",
        )
    }

    assert SEARCH_RESULT_TYPE_TEXT == "text"
    owner = sources["src/rag_core/search/vector_models.py"]
    assert owner.count('SEARCH_RESULT_TYPE_TEXT: Final[str] = "text"') == 1

    consumers = "\n".join(
        source
        for path, source in sources.items()
        if path != "src/rag_core/search/vector_models.py"
    )
    assert "SEARCH_RESULT_TYPE_TEXT" in consumers
    for duplicate in (
        'result.result_type != "text"',
        '"result_type": "text"',
        'result_type="text"',
    ):
        assert duplicate not in consumers



def test_search_result_payload_field_names_have_single_owner() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = {
        path: (root / path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/search/stored_payload_fields.py",
            "src/rag_core/search/result_filters.py",
        )
    }

    search_result_fields = {field.name for field in fields(make_search_result())}
    assert set(SEARCH_RESULT_FILTER_FIELDS).issubset(search_result_fields)
    assert set(SEARCH_RESULT_STORED_METADATA_FIELDS).issubset(
        set(SEARCH_RESULT_FILTER_FIELDS)
    )
    assert SEARCH_RESULT_LEGACY_THUMBNAIL_FIELD == "thumbnail_url"
    assert SEARCH_RESULT_LEGACY_THUMBNAIL_FIELD not in SEARCH_RESULT_FILTER_FIELDS
    assert SEARCH_RESULT_LEGACY_THUMBNAIL_FIELD not in search_result_fields

    owner = sources["src/rag_core/search/stored_payload_fields.py"]
    for name in (
        "SEARCH_RESULT_STORED_METADATA_FIELDS",
        "SEARCH_RESULT_FILTER_FIELDS",
        "SEARCH_RESULT_LEGACY_THUMBNAIL_FIELD",
    ):
        assert owner.count(f"{name}: Final") == 1

    consumer = sources["src/rag_core/search/result_filters.py"]
    assert "SEARCH_RESULT_FILTER_FIELDS" in consumer
    assert "SEARCH_RESULT_STORED_METADATA_FIELDS" not in consumer
    assert "SEARCH_RESULT_LEGACY_THUMBNAIL_FIELD" not in consumer
    assert "_RESULT_FILTER_FIELDS =" not in consumer
