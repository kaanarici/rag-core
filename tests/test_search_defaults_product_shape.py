"""Single-owner invariants for retrieval defaults and stored-payload field names.

The product contract is that each retrieval default / channel label / payload
field name lives in exactly one owner module and reaches consumers as a named
import (so changing a default is a one-line edit). These checks assert that via
the package import graph and top-level definition ownership, so they survive file
merges, renames, and reformatting. (Previously the same intent was asserted by
scraping a hand-pinned list of source files for exact literal substrings and
``count(...) == 1``, which froze the file layout and rewarded inlined literals.)

Value assertions (``DEFAULT == 10``) stay inline; round-trips stay inline.
"""

from __future__ import annotations

from dataclasses import fields

from rag_core.local_search.models import (
    DEFAULT_LOCAL_SEARCH_COLLECTION,
    DEFAULT_LOCAL_SEARCH_NAMESPACE,
)
from rag_core.local_search.runner import local_search_hit_payload
from rag_core.cli.output import search_hit_payload
from rag_core.retrieval_channels import (
    DENSE_RETRIEVAL_CHANNEL,
    RETRIEVAL_CHANNELS,
    SPARSE_RETRIEVAL_CHANNEL,
)
from rag_core.retrieval_defaults import (
    DEFAULT_CONTEXT_LIMIT,
    DEFAULT_LOCAL_SEARCH_LIMIT,
    DEFAULT_RERANK,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_USE_LEXICAL_SEARCH,
)
from rag_core.search import DenseChannel, Prefetch, QueryPlan, default_query_plan
from rag_core.search.pipeline_runner import SearchRequest
from rag_core.search.request_models import SearchQuery, SearchSidecarQuery
from rag_core.search.stored_payload_fields import (
    SEARCH_RESULT_FILTER_FIELDS,
    SEARCH_RESULT_LEGACY_THUMBNAIL_FIELD,
    SEARCH_RESULT_STORED_METADATA_FIELDS,
)
from rag_core.search.vector_models import (
    SEARCH_RESULT_TYPE_TEXT,
    SparseVector,
)
from tests.support import make_search_result
from tests.support.source_graph import (
    defining_modules,
    modules_importing,
    symbol_module,
)


def _importers_of(*roots: str, name: str) -> dict[str, list[str]]:
    return modules_importing(
        *roots,
        predicate=lambda module: module.rsplit(".", 1)[-1] == name,
    )


def test_lexical_search_default_has_single_retrieval_owner() -> None:
    assert DEFAULT_USE_LEXICAL_SEARCH is True
    assert defining_modules("src/rag_core", name="DEFAULT_USE_LEXICAL_SEARCH") == {
        "rag_core.retrieval_defaults"
    }
    # The default must reach engine/facade/runtime/contracts/pipeline as the named
    # import rather than being re-inlined per layer.
    importers = _importers_of("src/rag_core", name="DEFAULT_USE_LEXICAL_SEARCH")
    assert "rag_core.contracts.tool_contract_schemas" in importers
    assert {
        "rag_core._engine.core_retrieval",
        "rag_core.facade.retrieval",
        "rag_core.runtime.requests",
        "rag_core.search.pipeline_runner",
    } <= set(importers)


def test_rerank_default_has_single_retrieval_owner() -> None:
    assert DEFAULT_RERANK is False
    assert defining_modules("src/rag_core", name="DEFAULT_RERANK") == {
        "rag_core.retrieval_defaults"
    }
    importers = _importers_of("src/rag_core", name="DEFAULT_RERANK")
    assert "rag_core.contracts.tool_contract_schemas" in importers
    assert {
        "rag_core._engine.core_retrieval",
        "rag_core.facade.retrieval",
        "rag_core.runtime.requests",
        "rag_core.search.pipeline_runner",
    } <= set(importers)


def test_local_search_runner_does_not_introduce_a_parallel_search_plan() -> None:
    # The local-search runner reuses the engine run-spec; it must not grow a second
    # planning surface. Asserting on the spec symbols (and their absence) survives
    # file moves better than a literal scan, while still failing on a parallel plan.
    run_spec_owners = defining_modules(
        "src/rag_core/local_search", name="LocalSearchRunSpec"
    )
    assert run_spec_owners != set()
    builder_owners = defining_modules(
        "src/rag_core/local_search", name="build_local_search_run_spec"
    )
    assert builder_owners != set()
    assert (
        defining_modules("src/rag_core/local_search", name="LocalSearchPlan")
        == set()
    )
    assert (
        defining_modules(
            "src/rag_core/local_search", name="build_local_search_plan"
        )
        == set()
    )


def test_local_search_hit_payload_names_local_projection() -> None:
    # Two distinct payload projections with one owner each: the shared CLI payload
    # lives in cli.output; the local-search projection lives in the local runner
    # and wraps it. Asserting ownership (not call sites) survives refactors.
    assert symbol_module(search_hit_payload) == "rag_core.cli.output"
    assert (
        symbol_module(local_search_hit_payload)
        == "rag_core.local_search.runner"
    )
    assert (
        defining_modules("src/rag_core/local_search", name="search_hit_payload")
        == set()
    )
    assert (
        defining_modules("src/rag_core/ingest/local", name="search_hit_payload")
        == set()
    )
    # The local runner is the single owner of the local projection.
    assert defining_modules(
        "src/rag_core", name="local_search_hit_payload"
    ) == {"rag_core.local_search.runner"}


def test_public_entrypoint_defaults_are_named_once() -> None:
    assert DEFAULT_SEARCH_LIMIT == 10
    assert DEFAULT_LOCAL_SEARCH_LIMIT == 5
    assert DEFAULT_CONTEXT_LIMIT == 8
    for name, owner in (
        ("DEFAULT_SEARCH_LIMIT", "rag_core.retrieval_defaults"),
        ("DEFAULT_LOCAL_SEARCH_LIMIT", "rag_core.retrieval_defaults"),
        ("DEFAULT_CONTEXT_LIMIT", "rag_core.retrieval_defaults"),
    ):
        assert defining_modules("src/rag_core", name=name) == {owner}


def test_local_search_string_defaults_have_single_owner() -> None:
    assert DEFAULT_LOCAL_SEARCH_COLLECTION == "local_search"
    assert DEFAULT_LOCAL_SEARCH_NAMESPACE == "local"
    for name in ("DEFAULT_LOCAL_SEARCH_COLLECTION", "DEFAULT_LOCAL_SEARCH_NAMESPACE"):
        assert defining_modules("src/rag_core", name=name) == {
            "rag_core.local_search.models"
        }


def test_internal_search_pipeline_defaults_use_named_search_limit() -> None:
    # Every retrieval entrypoint defaults its limit to the one named constant.
    assert SearchRequest(query="q", collections=["c"], namespace="n").limit == (
        DEFAULT_SEARCH_LIMIT
    )
    assert (
        SearchQuery(
            dense_vector=[1.0],
            sparse_vector=SparseVector(indices=[1], values=[1.0]),
            namespace="n",
            collections=["c"],
        ).limit
        == DEFAULT_SEARCH_LIMIT
    )
    assert (
        SearchSidecarQuery(query="q", namespace="n", collections=["c"]).limit
        == DEFAULT_SEARCH_LIMIT
    )
    assert (
        QueryPlan(prefetches=(Prefetch(channel=DenseChannel(), limit=10),)).final_limit
        == DEFAULT_SEARCH_LIMIT
    )
    assert default_query_plan().final_limit == DEFAULT_SEARCH_LIMIT
    # Single owner, and the pipeline/provider layers consume it by name.
    assert defining_modules("src/rag_core", name="DEFAULT_SEARCH_LIMIT") == {
        "rag_core.retrieval_defaults"
    }
    importers = set(_importers_of("src/rag_core/search", name="DEFAULT_SEARCH_LIMIT"))
    assert {
        "rag_core.search.pipeline_runner",
        "rag_core.search.request_models",
    } <= importers


def test_retrieval_channel_labels_have_single_owner() -> None:
    assert DENSE_RETRIEVAL_CHANNEL == "dense"
    assert SPARSE_RETRIEVAL_CHANNEL == "sparse"
    assert RETRIEVAL_CHANNELS == ("dense", "sparse")
    for name in (
        "DENSE_RETRIEVAL_CHANNEL",
        "SPARSE_RETRIEVAL_CHANNEL",
        "RETRIEVAL_CHANNELS",
    ):
        assert defining_modules("src/rag_core", name=name) == {
            "rag_core.retrieval_channels"
        }
    # Channel-using layers (events, doctor output, retrieval/indexing) import the
    # labels by name rather than inlining "dense"/"sparse".
    importers = set(
        _importers_of("src/rag_core", name="DENSE_RETRIEVAL_CHANNEL")
    ) | set(_importers_of("src/rag_core", name="SPARSE_RETRIEVAL_CHANNEL"))
    assert {
        "rag_core.cli.doctor_output",
        "rag_core.events.document_events",
        "rag_core.search.pipeline.stages.hybrid_retrieve",
    } <= importers


def test_search_result_type_text_label_has_single_owner() -> None:
    assert SEARCH_RESULT_TYPE_TEXT == "text"
    assert defining_modules("src/rag_core/search", name="SEARCH_RESULT_TYPE_TEXT") == {
        "rag_core.search.vector_models"
    }
    importers = set(
        _importers_of("src/rag_core/search", name="SEARCH_RESULT_TYPE_TEXT")
    )
    assert {
        "rag_core.search.context_pack_helpers",
        "rag_core.search.stored_payload",
    } <= importers


def test_search_result_payload_field_names_have_single_owner() -> None:
    search_result_fields = {field.name for field in fields(make_search_result())}
    assert set(SEARCH_RESULT_FILTER_FIELDS).issubset(search_result_fields)
    assert set(SEARCH_RESULT_STORED_METADATA_FIELDS).issubset(
        set(SEARCH_RESULT_FILTER_FIELDS)
    )
    assert SEARCH_RESULT_LEGACY_THUMBNAIL_FIELD == "thumbnail_url"
    assert SEARCH_RESULT_LEGACY_THUMBNAIL_FIELD not in SEARCH_RESULT_FILTER_FIELDS
    assert SEARCH_RESULT_LEGACY_THUMBNAIL_FIELD not in search_result_fields

    for name in (
        "SEARCH_RESULT_STORED_METADATA_FIELDS",
        "SEARCH_RESULT_FILTER_FIELDS",
        "SEARCH_RESULT_LEGACY_THUMBNAIL_FIELD",
    ):
        assert defining_modules("src/rag_core/search", name=name) == {
            "rag_core.search.stored_payload_fields"
        }

    # result_filters consumes the filter set by name and does not re-derive the
    # other two field lists locally.
    assert "rag_core.search.result_filters" in _importers_of(
        "src/rag_core/search", name="SEARCH_RESULT_FILTER_FIELDS"
    )
    assert (
        "rag_core.search.result_filters"
        not in _importers_of(
            "src/rag_core/search", name="SEARCH_RESULT_STORED_METADATA_FIELDS"
        )
    )
