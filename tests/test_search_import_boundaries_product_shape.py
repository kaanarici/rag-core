from __future__ import annotations

from pathlib import Path



def test_context_pack_modules_import_search_result_owner_directly() -> None:
    root = Path(__file__).resolve().parents[1]
    context_pack_sources = {
        path: (root / path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/search/context_pack.py",
            "src/rag_core/search/context_pack_helpers.py",
            "src/rag_core/search/context_pack_sources.py",
        )
    }

    for path, source in context_pack_sources.items():
        assert "from rag_core.search.types import" not in source, path
        assert "rag_core.search.vector_models" in source, path



def test_indexer_modules_import_search_contract_owners_directly() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = {
        path: (root / path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/search/indexer.py",
            "src/rag_core/search/indexer_embeddings.py",
            "src/rag_core/search/indexer_embedding_vectors.py",
            "src/rag_core/search/indexer_points.py",
            "src/rag_core/search/indexer_texts.py",
            "src/rag_core/search/indexer_validation.py",
            "src/rag_core/search/text_builder.py",
        )
    }

    for path, source in sources.items():
        assert "from rag_core.search.types import" not in source, path
    for path in (
        "src/rag_core/search/indexer.py",
        "src/rag_core/search/indexer_embeddings.py",
        "src/rag_core/search/indexer_embedding_vectors.py",
        "src/rag_core/search/indexer_validation.py",
    ):
        assert "rag_core.search.provider_protocols" in sources[path], path
    assert "rag_core.search.request_models" in sources[
        "src/rag_core/search/indexer.py"
    ]
    for path in (
        "src/rag_core/search/indexer_embeddings.py",
        "src/rag_core/search/indexer_embedding_vectors.py",
        "src/rag_core/search/indexer_points.py",
        "src/rag_core/search/indexer_texts.py",
        "src/rag_core/search/text_builder.py",
    ):
        assert "rag_core.search.vector_models" in sources[path], path



def test_pipeline_modules_import_search_contract_owners_directly() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = {
        path: (root / path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/search/pipeline/types.py",
            "src/rag_core/search/pipeline/runner.py",
            "src/rag_core/search/pipeline/merge_strategies.py",
            "src/rag_core/search/pipeline_runner.py",
            "src/rag_core/search/pipeline/stages/hybrid_retrieve.py",
            "src/rag_core/search/pipeline/stages/identity.py",
            "src/rag_core/search/pipeline/stages/reranker_results.py",
            "src/rag_core/search/pipeline/stages/reranker_stage.py",
            "src/rag_core/search/pipeline/stages/reranker_stage_runtime.py",
            "src/rag_core/search/pipeline/stages/sidecar_application.py",
            "src/rag_core/search/pipeline/stages/sidecar_postprocess.py",
            "src/rag_core/search/pipeline/stages/sidecar_prefetch.py",
        )
    }

    for path, source in sources.items():
        assert "from rag_core.search.types import" not in source, path
    for path in (
        "src/rag_core/search/pipeline/types.py",
        "src/rag_core/search/pipeline_runner.py",
        "src/rag_core/search/pipeline/stages/hybrid_retrieve.py",
        "src/rag_core/search/pipeline/stages/reranker_stage_runtime.py",
        "src/rag_core/search/pipeline/stages/sidecar_prefetch.py",
    ):
        assert "rag_core.search.provider_protocols" in sources[path], path
    for path in (
        "src/rag_core/search/pipeline/types.py",
        "src/rag_core/search/pipeline_runner.py",
    ):
        assert "rag_core.search.filters" in sources[path], path
    for path in (
        "src/rag_core/search/pipeline/types.py",
        "src/rag_core/search/pipeline_runner.py",
        "src/rag_core/search/pipeline/stages/hybrid_retrieve.py",
        "src/rag_core/search/pipeline/stages/reranker_results.py",
        "src/rag_core/search/pipeline/stages/reranker_stage.py",
        "src/rag_core/search/pipeline/stages/reranker_stage_runtime.py",
        "src/rag_core/search/pipeline/stages/sidecar_application.py",
    ):
        assert "rag_core.search.request_models" in sources[path], path
    for path, source in sources.items():
        assert "rag_core.search.vector_models" in source, path



def test_search_helper_modules_import_contract_owners_directly() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = {
        path: (root / path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/search/embedding_cache_diagnostics.py",
            "src/rag_core/search/filter_eval.py",
            "src/rag_core/search/lexical_sidecar.py",
            "src/rag_core/search/lexical_sidecar_matching.py",
            "src/rag_core/search/query_plan_trace.py",
            "src/rag_core/search/result_filters.py",
            "src/rag_core/search/result_scores.py",
            "src/rag_core/search/stored_payload.py",
        )
    }

    for path, source in sources.items():
        assert "from rag_core.search.types import" not in source, path
    for path in (
        "src/rag_core/search/filter_eval.py",
        "src/rag_core/search/query_plan_trace.py",
    ):
        assert "rag_core.search.filters" in sources[path], path
    for path in (
        "src/rag_core/search/embedding_cache_diagnostics.py",
        "src/rag_core/search/lexical_sidecar.py",
        "src/rag_core/search/query_plan_trace.py",
    ):
        assert "rag_core.search.provider_protocols" in sources[path], path
    for path in (
        "src/rag_core/search/lexical_sidecar.py",
        "src/rag_core/search/lexical_sidecar_matching.py",
        "src/rag_core/search/query_plan_trace.py",
        "src/rag_core/search/result_filters.py",
    ):
        assert "rag_core.search.request_models" in sources[path], path
    for path in (
        "src/rag_core/search/lexical_sidecar.py",
        "src/rag_core/search/lexical_sidecar_matching.py",
        "src/rag_core/search/result_filters.py",
        "src/rag_core/search/result_scores.py",
        "src/rag_core/search/stored_payload.py",
    ):
        assert "rag_core.search.vector_models" in sources[path], path



def test_core_engine_modules_import_search_contract_owners_directly() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/core.py",
            "src/rag_core/core_assembly.py",
            "src/rag_core/core_builders.py",
            "src/rag_core/core_ingest.py",
            "src/rag_core/core_ingest_decision.py",
            "src/rag_core/core_ingest_delete.py",
            "src/rag_core/core_ingest_recovery.py",
            "src/rag_core/core_lifecycle.py",
            "src/rag_core/core_sidecar_sync.py",
            "src/rag_core/core_vector_store_factory.py",
            "src/rag_core/cli_doctor_store.py",
        )
    }

    provider_protocol_consumers = (
        "src/rag_core/core.py",
        "src/rag_core/core_assembly.py",
        "src/rag_core/core_ingest.py",
        "src/rag_core/core_ingest_decision.py",
        "src/rag_core/core_ingest_delete.py",
        "src/rag_core/core_ingest_recovery.py",
        "src/rag_core/core_sidecar_sync.py",
        "src/rag_core/core_vector_store_factory.py",
    )
    request_model_consumers = (
        "src/rag_core/core_builders.py",
        "src/rag_core/core_ingest_decision.py",
        "src/rag_core/core_ingest_recovery.py",
        "src/rag_core/core_lifecycle.py",
    )

    for path, source in sources.items():
        assert "from rag_core.search.types import" not in source, path
    for path in provider_protocol_consumers:
        assert "rag_core.search.provider_protocols" in sources[path]
    for path in request_model_consumers:
        assert "rag_core.search.request_models" in sources[path]
    assert "rag_core.search.vector_models" in sources["src/rag_core/cli_doctor_store.py"]
