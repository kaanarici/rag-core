from __future__ import annotations

from pathlib import Path

from rag_core.events.trace_payload_fields import (
    CONTEXT_PACK_SEARCH_STAGE,
    FUSE_SEARCH_STAGE,
    POSTPROCESS_SEARCH_STAGE,
    QUERY_TRANSFORM_SEARCH_STAGE,
    RERANK_SEARCH_STAGE,
    RETRIEVE_SEARCH_STAGE,
    SEARCH_ERROR_STAGE,
    TRACE_ABSENT_LABEL,
    TRACE_EMPTY_LABEL,
    TRACE_UNKNOWN_LABEL,
)

CANONICAL_LAUNCH_GATES = (
    "uv run ruff check .",
    "uv run mypy src tests examples",
    "uv run pytest -q",
    "./scripts/dx_smoke.sh",
    "./scripts/ci_self_host_smoke.sh",
    "uv build",
    "uv run python scripts/check_dist_artifacts.py",
    "uv run python scripts/wheel_smoke.py",
)


def test_trace_absent_labels_have_single_trace_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/events/trace_payload_fields.py",
            "src/rag_core/events/search_events.py",
            "src/rag_core/events/sink_field_policy.py",
            "src/rag_core/events/sink_payloads.py",
            "src/rag_core/search/query_plan_trace.py",
        )
    }

    assert TRACE_EMPTY_LABEL == ""
    assert TRACE_ABSENT_LABEL == "none"
    assert TRACE_UNKNOWN_LABEL == "unknown"
    trace_fields = sources["src/rag_core/events/trace_payload_fields.py"]
    assert trace_fields.count('TRACE_ABSENT_LABEL = "none"') == 1
    assert trace_fields.count('TRACE_EMPTY_LABEL = ""') == 1
    assert trace_fields.count('TRACE_UNKNOWN_LABEL = "unknown"') == 1
    assert 'return "unknown"' not in trace_fields
    assert 'return "none"' not in sources["src/rag_core/search/query_plan_trace.py"]
    assert (
        'QUERY_PLAN_TRACE_STORE_DEFAULT_LABEL = "store_default"'
        in sources["src/rag_core/search/query_plan_trace.py"]
    )
    assert (
        'return "store_default"'
        not in sources["src/rag_core/search/query_plan_trace.py"]
    )
    assert (
        'truncation_reason: str = "none"'
        not in sources["src/rag_core/events/search_events.py"]
    )
    for path in (
        "src/rag_core/events/search_events.py",
        "src/rag_core/events/sink_field_policy.py",
        "src/rag_core/search/query_plan_trace.py",
    ):
        assert "TRACE_ABSENT_LABEL" in sources[path]




def test_search_stage_labels_have_single_trace_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/events/trace_payload_fields.py",
            "src/rag_core/events/search_events.py",
            "src/rag_core/core_retrieval.py",
            "src/rag_core/search/pipeline/runner.py",
            "src/rag_core/search/pipeline_runner.py",
        )
    }

    expected = {
        "QUERY_TRANSFORM_SEARCH_STAGE": QUERY_TRANSFORM_SEARCH_STAGE,
        "RETRIEVE_SEARCH_STAGE": RETRIEVE_SEARCH_STAGE,
        "FUSE_SEARCH_STAGE": FUSE_SEARCH_STAGE,
        "RERANK_SEARCH_STAGE": RERANK_SEARCH_STAGE,
        "POSTPROCESS_SEARCH_STAGE": POSTPROCESS_SEARCH_STAGE,
        "CONTEXT_PACK_SEARCH_STAGE": CONTEXT_PACK_SEARCH_STAGE,
        "SEARCH_ERROR_STAGE": SEARCH_ERROR_STAGE,
    }
    assert expected == {
        "QUERY_TRANSFORM_SEARCH_STAGE": "query_transform",
        "RETRIEVE_SEARCH_STAGE": "retrieve",
        "FUSE_SEARCH_STAGE": "fuse",
        "RERANK_SEARCH_STAGE": "rerank",
        "POSTPROCESS_SEARCH_STAGE": "postprocess",
        "CONTEXT_PACK_SEARCH_STAGE": "context_pack",
        "SEARCH_ERROR_STAGE": "search",
    }

    owner = sources["src/rag_core/events/trace_payload_fields.py"]
    for symbol, value in expected.items():
        assert symbol in owner
        assert owner.count(f'"{value}"') >= 1

    consumers = "\n".join(
        source
        for path, source in sources.items()
        if path != "src/rag_core/events/trace_payload_fields.py"
    )
    for symbol in expected:
        assert symbol in consumers
    for value in expected.values():
        assert f'stage="{value}"' not in consumers
        assert f'stage: str = "{value}"' not in consumers
        assert f'_emit_stage_error(sink, stage="{value}"' not in consumers
