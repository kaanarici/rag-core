"""Single-owner invariants for query-plan kind labels, presets, and profiles.

Each fusion/boost kind label, query-plan preset, search profile, sparse-channel
name, and turbopuffer execution type must live in exactly one owner module and be
consumed by name (so the engine never hard-codes a second copy of a literal).
These checks assert that via top-level definition ownership and the package
import graph, so they survive file merges, renames, and reformatting. (Previously
the same intent was asserted by scraping a hand-pinned list of source files for
exact literal substrings and ``count(...) == 1``, which froze the file layout and
rewarded inlined string literals.)

Value assertions (``LABEL == "rrf"``) and dataclass-shape assertions stay inline.
"""

from __future__ import annotations

from dataclasses import fields

from rag_core.search.pipeline_runner import SearchExecutionOptions, SearchRequest
from rag_core.search.planning import (
    describe_search_profile_catalog as planning_describe_catalog,
)
from rag_core.search.query_plan import (
    BOOST_KIND_EXP_DECAY,
    BOOST_KIND_GAUSS_DECAY,
    BOOST_KIND_LINEAR_DECAY,
    BOOST_KIND_RAW,
    DEFAULT_RRF_K,
    FUSION_KIND_DBSF,
    FUSION_KIND_RRF,
    FUSION_KIND_WEIGHTED_RRF,
    PRIMARY_DENSE_QUERY_VECTOR,
)
from rag_core.search.query_plan_presets import (
    DEFAULT_SEARCH_PROFILE,
    QUERY_PLAN_PRESET_DENSE_ONLY,
    QUERY_PLAN_PRESET_HYBRID_DBSF,
    QUERY_PLAN_PRESET_HYBRID_RRF,
    QUERY_PLAN_PRESET_HYBRID_WITH_MMR,
    QUERY_PLAN_PRESET_SPARSE_ONLY,
    QUERY_PLAN_RERANK_MMR,
    SEARCH_PROFILE_BALANCED,
    SEARCH_PROFILE_COVERAGE,
    SEARCH_PROFILE_DIVERSE,
    SEARCH_PROFILE_FAST,
    SEARCH_PROFILE_LEXICAL,
)
from rag_core.search.sparse_channels import PRIMARY_SPARSE_CHANNEL
from tests.support.source_graph import (
    defining_modules,
    modules_importing,
    symbol_module,
    under_module,
)

_QUERY_PLAN = "rag_core.search.query_plan"
_PRESETS = "rag_core.search.query_plan_presets"
_TURBOPUFFER_PLAN = "rag_core.search.providers.turbopuffer_query"


def _importers_of(*roots: str, name: str) -> set[str]:
    return set(
        modules_importing(
            *roots,
            predicate=lambda module: module.rsplit(".", 1)[-1] == name,
        )
    )


def test_adjacent_search_request_shapes_name_their_layers() -> None:
    # The durable contract is that the engine-level request and the advanced
    # execution options are disjoint dataclasses with the documented field split,
    # so retrieval intent and runner controls cannot bleed into one another.
    request_fields = {field.name for field in fields(SearchRequest)}
    execution_fields = {field.name for field in fields(SearchExecutionOptions)}
    assert "execution" in request_fields
    assert execution_fields == {
        "query_vector",
        "query_sparse_vectors",
        "use_lexical_search",
        "query_plan",
    }
    assert not execution_fields & request_fields


def test_rrf_default_has_single_query_plan_owner() -> None:
    assert DEFAULT_RRF_K == 60
    assert defining_modules("src/rag_core/search", name="DEFAULT_RRF_K") == {_QUERY_PLAN}
    # Providers consume the named default rather than re-inlining ``60``.
    importers = _importers_of("src/rag_core/search", name="DEFAULT_RRF_K")
    assert {
        "rag_core.search.providers.memory_query_scoring",
        "rag_core.search.providers.qdrant_query",
    } <= importers


def test_query_plan_kind_labels_have_single_query_plan_owner() -> None:
    assert FUSION_KIND_RRF == "rrf"
    assert FUSION_KIND_DBSF == "dbsf"
    assert FUSION_KIND_WEIGHTED_RRF == "weighted_rrf"
    assert BOOST_KIND_LINEAR_DECAY == "linear_decay"
    assert BOOST_KIND_EXP_DECAY == "exp_decay"
    assert BOOST_KIND_GAUSS_DECAY == "gauss_decay"
    assert BOOST_KIND_RAW == "raw"

    labels = (
        "FUSION_KIND_RRF",
        "FUSION_KIND_DBSF",
        "FUSION_KIND_WEIGHTED_RRF",
        "BOOST_KIND_LINEAR_DECAY",
        "BOOST_KIND_EXP_DECAY",
        "BOOST_KIND_GAUSS_DECAY",
        "BOOST_KIND_RAW",
    )
    for label in labels:
        assert defining_modules("src/rag_core/search", name=label) == {_QUERY_PLAN}

    # Plan-consuming providers reference the labels by name. Asserting on the
    # qdrant provider (which branches on fusion/boost kind) keeps the "no inlined
    # literal" intent without pinning which files import which label.
    importers: set[str] = set()
    for label in labels:
        importers |= _importers_of("src/rag_core/search", name=label)
    assert "rag_core.search.providers.qdrant_query" in importers


def test_query_plan_preset_and_profile_names_have_single_owner() -> None:
    assert QUERY_PLAN_PRESET_HYBRID_RRF == "hybrid_rrf"
    assert QUERY_PLAN_PRESET_DENSE_ONLY == "dense_only"
    assert QUERY_PLAN_PRESET_SPARSE_ONLY == "sparse_only"
    assert QUERY_PLAN_PRESET_HYBRID_DBSF == "hybrid_dbsf"
    assert QUERY_PLAN_PRESET_HYBRID_WITH_MMR == "hybrid_with_mmr"
    assert QUERY_PLAN_RERANK_MMR == "mmr"
    assert SEARCH_PROFILE_BALANCED == "balanced"
    assert SEARCH_PROFILE_FAST == "fast"
    assert SEARCH_PROFILE_LEXICAL == "lexical"
    assert SEARCH_PROFILE_COVERAGE == "coverage"
    assert SEARCH_PROFILE_DIVERSE == "diverse"
    assert DEFAULT_SEARCH_PROFILE == SEARCH_PROFILE_BALANCED

    for label in (
        "QUERY_PLAN_PRESET_HYBRID_RRF",
        "QUERY_PLAN_PRESET_DENSE_ONLY",
        "QUERY_PLAN_PRESET_SPARSE_ONLY",
        "QUERY_PLAN_PRESET_HYBRID_DBSF",
        "QUERY_PLAN_PRESET_HYBRID_WITH_MMR",
        "QUERY_PLAN_RERANK_MMR",
        "SEARCH_PROFILE_BALANCED",
        "SEARCH_PROFILE_FAST",
        "SEARCH_PROFILE_LEXICAL",
        "SEARCH_PROFILE_COVERAGE",
        "SEARCH_PROFILE_DIVERSE",
    ):
        assert defining_modules("src/rag_core/search", name=label) == {_PRESETS}

    assert "rag_core.search.planning" in _importers_of(
        "src/rag_core/search", name="QUERY_PLAN_PRESET_DENSE_ONLY"
    )


def test_primary_sparse_channel_has_single_owner() -> None:
    assert PRIMARY_SPARSE_CHANNEL == "bm25"
    assert defining_modules("src/rag_core/search", name="PRIMARY_SPARSE_CHANNEL") == {
        "rag_core.search.sparse_channels"
    }
    importers = _importers_of("src/rag_core", name="PRIMARY_SPARSE_CHANNEL")
    assert {
        "rag_core.search.query_plan",
        "rag_core.search.query_plan_presets",
    } <= importers


def test_turbopuffer_search_execution_modes_have_single_owner() -> None:
    # The execution-type union replaced a stringly-typed mode enum. Asserting the
    # execution dataclasses have one owner and no mode enum/string survives
    # better than literal-substring scans.
    for execution in (
        "TurboPufferDenseExecution",
        "TurboPufferBm25Execution",
        "TurboPufferHybridRrfExecution",
    ):
        assert defining_modules("src/rag_core/search/providers", name=execution) == {
            _TURBOPUFFER_PLAN
        }
    assert (
        defining_modules("src/rag_core/search/providers", name="TurboPufferSearchMode")
        == set()
    )
    # The turbopuffer query module dispatches on the execution types from their
    # owner, not on a re-introduced ``TURBOPUFFER_SEARCH_MODE_*`` constant. The
    # store consumes the typed query-plan limit helper from that same owner.
    assert "rag_core.search.providers.turbopuffer_store" in _importers_of(
        "src/rag_core/search/providers", name="_supported_query_plan_limit"
    )
    assert (
        defining_modules(
            "src/rag_core/search/providers", name="TURBOPUFFER_SEARCH_MODE_SPARSE_KNN"
        )
        == set()
    )


def test_primary_dense_query_vector_has_single_query_plan_owner() -> None:
    assert PRIMARY_DENSE_QUERY_VECTOR == "primary"
    assert defining_modules(
        "src/rag_core/search", name="PRIMARY_DENSE_QUERY_VECTOR"
    ) == {_QUERY_PLAN}
    # Trace/validation/provider layers consume the named vector id.
    importers = _importers_of("src/rag_core/search", name="PRIMARY_DENSE_QUERY_VECTOR")
    assert {
        "rag_core.search.query_plan_trace",
        "rag_core.search.providers.memory_query_plan_validation",
        "rag_core.search.providers.turbopuffer_query",
    } <= importers


def test_query_plan_trace_helper_uses_query_plan_language() -> None:
    # The trace emitter is named in query-plan language and owned by the trace
    # module; the pipeline runner emits through it (the old ``emit_search_planned``
    # name must not return anywhere).
    from rag_core.search.query_plan_trace import emit_query_plan_trace_event

    assert (
        symbol_module(emit_query_plan_trace_event)
        == "rag_core.search.query_plan_trace"
    )
    assert "rag_core.search.pipeline_runner" in _importers_of(
        "src/rag_core/search", name="emit_query_plan_trace_event"
    )
    assert (
        defining_modules("src/rag_core/search", name="emit_search_planned") == set()
    )
    assert (
        defining_modules("src/rag_core/search", name="_EmitSearchPlannedTransform")
        == set()
    )


def test_runtime_search_diagnostics_use_public_planning_facade() -> None:
    # ``describe_search_profile_catalog`` is re-exported by the public planning
    # facade; runtime diagnostics must import it from the facade, not from the
    # internal presets module. (Where the symbol is *defined* is presets; the
    # contract is which surface consumers depend on.)
    assert planning_describe_catalog is not None
    assert (
        _importers_of(
            "src/rag_core/cli/commands",
            "src/rag_core/_engine",
            name="describe_search_profile_catalog",
        )
        >= {"rag_core.cli.commands.doctor", "rag_core._engine.core_runtime"}
    )
    presets_importers = modules_importing(
        "src/rag_core/cli/commands",
        "src/rag_core/_engine",
        predicate=under_module(_PRESETS),
    )
    assert presets_importers == {}
