from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from rag_core.search.pipeline_runner import SearchExecutionOptions, SearchRequest
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


def test_adjacent_search_request_shapes_name_their_layers() -> None:
    root = Path(__file__).resolve().parents[1]
    pipeline_runner = (
        root / "src" / "rag_core" / "search" / "pipeline_runner.py"
    ).read_text(encoding="utf-8")
    request_models = (
        root / "src" / "rag_core" / "search" / "request_models.py"
    ).read_text(encoding="utf-8")

    assert "Engine-level retrieval intent" in pipeline_runner
    assert "Advanced runner controls outside normal retrieval intent" in pipeline_runner
    assert "Vector-store execution query" in request_models
    assert "Sidecar execution query" in request_models

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
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/search/query_plan.py",
            "src/rag_core/search/query_plan_presets.py",
            "src/rag_core/search/providers/memory_query_scoring.py",
            "src/rag_core/search/providers/qdrant_query_plan.py",
        )
    }

    assert DEFAULT_RRF_K == 60
    assert sources["src/rag_core/search/query_plan.py"].count("DEFAULT_RRF_K = 60") == 1
    assert "rrf_k: int = DEFAULT_RRF_K" in sources["src/rag_core/search/query_plan.py"]
    assert (
        "_RRF_K = 60"
        not in sources["src/rag_core/search/providers/memory_query_scoring.py"]
    )
    assert (
        "DEFAULT_RRF_K + rank + 1"
        in sources["src/rag_core/search/providers/memory_query_scoring.py"]
    )
    assert (
        "fuse.rrf_k == DEFAULT_RRF_K"
        in sources["src/rag_core/search/providers/qdrant_query_plan.py"]
    )
    assert "rrf_k=60" not in sources["src/rag_core/search/query_plan_presets.py"]
    assert (
        "fuse.rrf_k == 60"
        not in sources["src/rag_core/search/providers/qdrant_query_plan.py"]
    )


def test_query_plan_kind_labels_have_single_query_plan_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/search/query_plan.py",
            "src/rag_core/search/query_plan_presets.py",
            "src/rag_core/search/planning.py",
            "src/rag_core/search/providers/memory_query_plan_validation.py",
            "src/rag_core/search/providers/qdrant_query_plan.py",
            "src/rag_core/search/providers/qdrant_store_guards.py",
            "src/rag_core/search/providers/turbopuffer_query_plan.py",
        )
    }

    assert FUSION_KIND_RRF == "rrf"
    assert FUSION_KIND_DBSF == "dbsf"
    assert FUSION_KIND_WEIGHTED_RRF == "weighted_rrf"
    assert BOOST_KIND_LINEAR_DECAY == "linear_decay"
    assert BOOST_KIND_EXP_DECAY == "exp_decay"
    assert BOOST_KIND_GAUSS_DECAY == "gauss_decay"
    assert BOOST_KIND_RAW == "raw"

    owner = sources["src/rag_core/search/query_plan.py"]
    assert owner.count('FUSION_KIND_RRF: Final[FusionKind] = "rrf"') == 1
    assert owner.count('FUSION_KIND_DBSF: Final[FusionKind] = "dbsf"') == 1
    assert (
        owner.count('FUSION_KIND_WEIGHTED_RRF: Final[FusionKind] = "weighted_rrf"') == 1
    )
    assert (
        owner.count('BOOST_KIND_LINEAR_DECAY: Final[BoostKind] = "linear_decay"') == 1
    )
    assert owner.count('BOOST_KIND_EXP_DECAY: Final[BoostKind] = "exp_decay"') == 1
    assert owner.count('BOOST_KIND_GAUSS_DECAY: Final[BoostKind] = "gauss_decay"') == 1
    assert owner.count('BOOST_KIND_RAW: Final[BoostKind] = "raw"') == 1
    assert "kind: FusionKind = FUSION_KIND_RRF" in owner
    assert "kind: BoostKind" in owner

    consumers = "\n".join(
        source
        for path, source in sources.items()
        if path != "src/rag_core/search/query_plan.py"
    )
    for symbol in (
        "FUSION_KIND_RRF",
        "FUSION_KIND_DBSF",
        "FUSION_KIND_WEIGHTED_RRF",
        "BOOST_KIND_LINEAR_DECAY",
        "BOOST_KIND_EXP_DECAY",
        "BOOST_KIND_GAUSS_DECAY",
        "BOOST_KIND_RAW",
    ):
        assert symbol in consumers
    for duplicate in (
        'fusion="rrf"',
        'fusion="dbsf"',
        'fusion="weighted_rrf"',
        'fusion: FusionKind = "rrf"',
        'fusion in ("rrf", "weighted_rrf")',
        'fuse.kind == "rrf"',
        'fuse.kind == "dbsf"',
        'fuse.kind == "weighted_rrf"',
        'fuse.kind != "rrf"',
        'boost.kind == "raw"',
        'boost.kind == "linear_decay"',
        'boost.kind == "exp_decay"',
        'boost.kind == "gauss_decay"',
    ):
        assert duplicate not in consumers


def test_query_plan_preset_and_profile_names_have_single_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/search/query_plan_presets.py",
            "src/rag_core/search/planning.py",
            "src/rag_core/search/providers/qdrant_search.py",
            "src/rag_core/search/pipeline/stages/hybrid_retrieve.py",
        )
    }

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

    owner = sources["src/rag_core/search/query_plan_presets.py"]
    for definition in (
        'QUERY_PLAN_PRESET_HYBRID_RRF: Final[str] = "hybrid_rrf"',
        'QUERY_PLAN_PRESET_DENSE_ONLY: Final[str] = "dense_only"',
        'QUERY_PLAN_PRESET_SPARSE_ONLY: Final[str] = "sparse_only"',
        'QUERY_PLAN_PRESET_HYBRID_DBSF: Final[str] = "hybrid_dbsf"',
        'QUERY_PLAN_PRESET_HYBRID_WITH_MMR: Final[str] = "hybrid_with_mmr"',
        'QUERY_PLAN_RERANK_MMR: Final[str] = "mmr"',
        'SEARCH_PROFILE_BALANCED: Final[str] = "balanced"',
        'SEARCH_PROFILE_FAST: Final[str] = "fast"',
        'SEARCH_PROFILE_LEXICAL: Final[str] = "lexical"',
        'SEARCH_PROFILE_COVERAGE: Final[str] = "coverage"',
        'SEARCH_PROFILE_DIVERSE: Final[str] = "diverse"',
    ):
        assert owner.count(definition) == 1
    assert "DEFAULT_SEARCH_PROFILE = SEARCH_PROFILE_BALANCED" in owner

    consumers = "\n".join(
        source
        for path, source in sources.items()
        if path != "src/rag_core/search/query_plan_presets.py"
    )
    assert "QUERY_PLAN_PRESET_DENSE_ONLY" in consumers
    for duplicate in (
        'query_plan_preset("dense_only"',
        'query_plan_preset("sparse_only"',
        'preset="hybrid_rrf"',
        'preset="dense_only"',
        'preset="sparse_only"',
        'preset="hybrid_dbsf"',
        'preset="hybrid_with_mmr"',
    ):
        assert duplicate not in consumers


def test_primary_sparse_channel_has_single_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/demo.py",
            "src/rag_core/search/sparse_channels.py",
            "src/rag_core/search/query_plan.py",
            "src/rag_core/search/query_plan_presets.py",
            "src/rag_core/search/providers/provider_category_diagnostics.py",
        )
    }

    owner = sources["src/rag_core/search/sparse_channels.py"]
    assert PRIMARY_SPARSE_CHANNEL == "bm25"
    assert owner.count('PRIMARY_SPARSE_CHANNEL = "bm25"') == 1
    for path in (
        "src/rag_core/demo.py",
        "src/rag_core/search/query_plan.py",
        "src/rag_core/search/query_plan_presets.py",
        "src/rag_core/search/providers/provider_category_diagnostics.py",
    ):
        assert "PRIMARY_SPARSE_CHANNEL" in sources[path]
    assert (
        'vector_field: str = "bm25"' not in sources["src/rag_core/search/query_plan.py"]
    )
    assert (
        'using_query_vector: str = "bm25"'
        not in sources["src/rag_core/search/query_plan.py"]
    )
    assert '"bm25":' not in sources["src/rag_core/demo.py"]
    assert (
        '"bm25": {'
        not in sources["src/rag_core/search/providers/provider_category_diagnostics.py"]
    )


def test_turbopuffer_search_execution_modes_have_single_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/search/providers/turbopuffer_query_plan.py",
            "src/rag_core/search/providers/turbopuffer_search.py",
        )
    }

    owner = sources["src/rag_core/search/providers/turbopuffer_query_plan.py"]
    for execution in (
        "TurboPufferDenseExecution",
        "TurboPufferBm25Execution",
        "TurboPufferHybridRrfExecution",
    ):
        assert execution in owner
    assert "TurboPufferSearchMode" not in owner
    assert "TURBOPUFFER_SEARCH_MODE_" not in owner

    consumer = sources["src/rag_core/search/providers/turbopuffer_search.py"]
    for execution in (
        "TurboPufferDenseExecution",
        "TurboPufferBm25Execution",
        "TurboPufferHybridRrfExecution",
    ):
        assert execution in consumer
    assert "TURBOPUFFER_SEARCH_MODE_SPARSE_KNN" not in consumer
    assert "TURBOPUFFER_SEARCH_MODE_" not in consumer
    assert "execution.mode" not in consumer
    assert 'execution.mode == "dense"' not in consumer
    assert 'execution.mode == "sparse_knn"' not in consumer
    assert 'execution.mode == "bm25"' not in consumer
    assert 'execution.mode == "hybrid_rrf"' not in consumer
    assert 'mode="dense"' not in consumer
    assert 'mode="bm25"' not in consumer
    assert 'mode="hybrid_rrf"' not in consumer
    assert 'mode="sparse_knn"' not in consumer


def test_primary_dense_query_vector_has_single_query_plan_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/search/query_plan.py",
            "src/rag_core/search/query_plan_trace.py",
            "src/rag_core/search/providers/memory_query_plan_validation.py",
            "src/rag_core/search/providers/turbopuffer_query_plan.py",
            "tests/test_event_trace_summary.py",
            "tests/test_events.py",
            "tests/test_events_opentelemetry.py",
            "tests/test_default_search_pipeline.py",
            "tests/test_capability_aware_default_query_plan.py",
        )
    }

    owner = sources["src/rag_core/search/query_plan.py"]
    assert PRIMARY_DENSE_QUERY_VECTOR == "primary"
    assert owner.count('PRIMARY_DENSE_QUERY_VECTOR: Final[str] = "primary"') == 1
    assert "using_query_vector: str = PRIMARY_DENSE_QUERY_VECTOR" in owner
    for path, source in sources.items():
        if path == "src/rag_core/search/query_plan.py":
            continue
        assert "PRIMARY_DENSE_QUERY_VECTOR" in source
        assert '!= "primary"' not in source
        assert 'or "primary"' not in source
        assert "dense:dense:primary" not in source


def test_query_plan_trace_helper_uses_query_plan_language() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/search/query_plan_trace.py",
            "src/rag_core/search/pipeline_runner.py",
            "tests/test_events.py",
        )
    }

    assert (
        "def emit_query_plan_trace_event("
        in sources["src/rag_core/search/query_plan_trace.py"]
    )
    assert (
        "emit_query_plan_trace_event("
        in sources["src/rag_core/search/pipeline_runner.py"]
    )
    assert "emit_query_plan_trace_event(" in sources["tests/test_events.py"]
    for source in sources.values():
        assert "emit_search_planned" not in source
        assert "_EmitSearchPlannedTransform" not in source


def test_runtime_search_diagnostics_use_public_planning_facade() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/cli_doctor.py",
            "src/rag_core/_engine/core_runtime.py",
        )
    }

    for source in sources.values():
        assert (
            "from rag_core.search.planning import describe_search_profile_catalog"
            in source
        )
        assert (
            "from rag_core.search.query_plan_presets import "
            "describe_search_profile_catalog" not in source
        )
