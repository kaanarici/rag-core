from __future__ import annotations

from typing import cast

import rag_core.events as events
from rag_core.events import (
    summarize_search_trace,
    summarize_search_trace_payloads,
    summarize_search_trace_payload_runs,
    summarize_search_trace_runs,
)
from rag_core.events.trace_payload_fields import TRACE_ABSENT_LABEL
from rag_core.events.types import (
    RerankApplied,
    SearchCompleted,
    SearchPlanned,
    SearchStarted,
    SearchStageCompleted,
    SidecarApplied,
    StageError,
)
from rag_core.search.query_plan import PRIMARY_DENSE_QUERY_VECTOR

_DENSE_PRIMARY_CHANNEL = f"dense:dense:{PRIMARY_DENSE_QUERY_VECTOR}"


def test_summarize_search_trace_exports_safe_app_facing_shape() -> None:
    assert "summarize_search_trace" in events.__all__
    assert "summarize_search_trace_payloads" in events.__all__

    summary = summarize_search_trace(
        [
            SearchStarted(
                namespace="tenant-secret",
                corpus_ids=("corpus-secret",),
                query_length=len("private billing query"),
                limit=5,
            ),
            SearchPlanned(
                namespace="tenant-secret",
                corpus_ids=("corpus-secret",),
                limit=5,
                final_limit=4,
                channels=(_DENSE_PRIMARY_CHANNEL, "sparse:bm25:bm25"),
                prefetch_limits=(10, 20),
                search_profile="balanced",
                fusion="rrf",
                plan_rerank="provider",
                metadata_filter="Term",
                rerank_candidate_count=8,
                rerank_timeout_ms=250.0,
                rerank_max_output=4,
                rerank_fallback_on_error=False,
                use_lexical_search=True,
                retrieve_stage="HybridRetrieve",
                fuse_stage="ReciprocalRankFusion",
                rerank_stage="ProviderRerank",
                postprocesses=("SidecarPostprocess",),
            ),
            SearchStageCompleted(
                stage="retrieve",
                stage_name="HybridRetrieve",
                candidate_count=10,
                result_count=3,
                duration_ms=1.5,
            ),
            SearchStageCompleted(
                stage="context_pack",
                stage_name="build_context_pack",
                candidate_count=3,
                result_count=2,
                dropped_count=1,
                truncated=True,
                max_chars=256,
                max_tokens=64,
                token_estimate=42,
                char_count=168,
                citation_count=2,
                source_preview_count=2,
                duration_ms=0.2,
            ),
            RerankApplied(
                provider="cohere",
                model="rerank-v3.5",
                input_count=10,
                candidate_count=8,
                result_count=4,
                top_k=4,
                fallback_reason="TimeoutError",
                truncation_reason="candidate_budget",
                duration_ms=3.5,
                succeeded=False,
                provider_result_count=6,
                accepted_count=4,
                dropped_count=2,
                rank_changed_count=3,
                rank_promoted_count=2,
                rank_demoted_count=1,
                max_rank_gain=3,
                max_rank_loss=1,
                provider_score_min=0.12,
                provider_score_max=0.98,
                search_score_min=0.21,
                search_score_max=0.87,
            ),
            SidecarApplied(
                provider="bm25",
                input_count=5,
                provider_result_count=7,
                accepted_count=3,
                dropped_count=4,
                result_count=3,
                duration_ms=1.25,
                succeeded=True,
            ),
            StageError(
                stage="rerank",
                error_type="RuntimeError",
                message="raw provider detail for /private/doc-1.txt",
            ),
            SearchCompleted(
                namespace="tenant-secret",
                result_count=2,
                duration_ms=12.0,
            ),
        ]
    )

    payload = summary.to_payload()

    assert payload["query_length"] == len("private billing query")
    assert payload["corpus_count"] == 1
    assert payload["channels"] == [_DENSE_PRIMARY_CHANNEL, "sparse:bm25:bm25"]
    assert payload["search_profile"] == "balanced"
    assert payload["rerank_fallback_on_error"] is False
    assert payload["attempted_rerank"] is True
    assert payload["applied_rerank"] is False
    assert payload["rerank_provider"] == "cohere"
    assert payload["rerank_model"] == "rerank-v3.5"
    assert payload["rerank_applied_candidate_count"] == 8
    assert payload["rerank_provider_result_count"] == 6
    assert payload["rerank_accepted_count"] == 4
    assert payload["rerank_dropped_count"] == 2
    assert payload["rerank_rank_changed_count"] == 3
    assert payload["rerank_rank_promoted_count"] == 2
    assert payload["rerank_rank_demoted_count"] == 1
    assert payload["rerank_max_rank_gain"] == 3
    assert payload["rerank_max_rank_loss"] == 1
    assert payload["rerank_provider_score_min"] == 0.12
    assert payload["rerank_provider_score_max"] == 0.98
    assert payload["rerank_search_score_min"] == 0.21
    assert payload["rerank_search_score_max"] == 0.87
    assert payload["rerank_result_count"] == 4
    assert payload["rerank_top_k"] == 4
    assert payload["rerank_fallback_reason"] == "TimeoutError"
    assert payload["rerank_truncation_reason"] == "candidate_budget"
    assert payload["rerank_succeeded"] is False
    assert payload["applied_sidecar"] is True
    assert payload["attempted_sidecar"] is True
    assert payload["sidecar_provider"] == "bm25"
    assert payload["sidecar_input_count"] == 5
    assert payload["sidecar_provider_result_count"] == 7
    assert payload["sidecar_accepted_count"] == 3
    assert payload["sidecar_dropped_count"] == 4
    assert payload["sidecar_result_count"] == 3
    assert payload["sidecar_succeeded"] is True
    assert payload["stage_count"] == 2
    assert payload["completed"] is True
    assert payload["error_count"] == 1
    assert payload["error_stages"] == ["rerank"]
    assert payload["error_types"] == ["RuntimeError"]
    stages = cast(list[dict[str, object]], payload["stages"])
    context_pack_stage = stages[1]
    assert isinstance(context_pack_stage, dict)
    assert context_pack_stage["stage"] == "context_pack"
    assert context_pack_stage["dropped_count"] == 1
    assert context_pack_stage["truncated"] is True
    assert context_pack_stage["citation_count"] == 2
    assert context_pack_stage["source_preview_count"] == 2

    rendered_payload = repr(payload)
    assert "private billing query" not in rendered_payload
    assert "tenant-secret" not in rendered_payload
    assert "corpus-secret" not in rendered_payload
    assert "/private/doc-1.txt" not in rendered_payload
    assert "raw provider detail" not in rendered_payload


def test_rerank_summary_applied_requires_accepted_results() -> None:
    summary = summarize_search_trace(
        [
            RerankApplied(
                provider="cohere",
                model="rerank-v3.5",
                input_count=2,
                candidate_count=2,
                provider_result_count=0,
                accepted_count=0,
                dropped_count=0,
                result_count=2,
                top_k=2,
                succeeded=True,
            ),
            SearchCompleted(
                result_count=2,
                requested_rerank=True,
                attempted_rerank=True,
                applied_rerank=False,
            ),
        ]
    )

    payload = summary.to_payload()
    assert payload["rerank_succeeded"] is True
    assert payload["applied_rerank"] is False


def test_summarize_search_trace_payloads_preserves_unknown_rerank_scores() -> None:
    summary = summarize_search_trace_payloads(
        [
            {
                "event_type": "rerank.applied",
                "provider": "cohere",
                "model": "rerank-v3.5",
                "accepted_count": 2,
                "succeeded": True,
            }
        ]
    )

    payload = summary.to_payload()
    assert payload["rerank_accepted_count"] == 2
    assert payload["rerank_provider_score_min"] is None
    assert payload["rerank_provider_score_max"] is None
    assert payload["rerank_search_score_min"] is None
    assert payload["rerank_search_score_max"] is None


def test_search_completed_rerank_and_sidecar_status_uses_completion_fields() -> None:
    summary = summarize_search_trace(
        [
            SearchCompleted(
                result_count=2,
                requested_rerank=True,
                requested_sidecar=True,
                attempted_rerank=True,
                attempted_sidecar=True,
                applied_rerank=True,
                applied_sidecar=True,
            )
        ]
    )

    payload = summary.to_payload()
    assert payload["attempted_rerank"] is True
    assert payload["attempted_sidecar"] is True
    assert payload["applied_rerank"] is True
    assert payload["applied_sidecar"] is True
    assert "used_rerank" not in payload
    assert "used_sidecar" not in payload
    assert "rerank_attempted" not in payload
    assert "sidecar_attempted" not in payload


def test_trace_payload_summary_ignores_removed_legacy_keys() -> None:
    summary = summarize_search_trace_payloads(
        [
            {
                "event_type": "search.planned",
                "use_sidecar": True,
            },
            {
                "event_type": "search.completed",
                "result_count": 1,
                "rerank_attempted": True,
                "sidecar_attempted": True,
            },
        ]
    )

    payload = summary.to_payload()
    assert payload["use_lexical_search"] is False
    assert payload["attempted_rerank"] is False
    assert payload["attempted_sidecar"] is False
    assert "rerank_attempted" not in payload
    assert "sidecar_attempted" not in payload


def test_failed_search_completion_is_terminal_but_not_successful() -> None:
    summary = summarize_search_trace(
        [
            SearchStarted(query_length=12, limit=5),
            StageError(stage="search", error_type="RuntimeError"),
            SearchCompleted(
                result_count=0,
                requested_rerank=True,
                requested_sidecar=True,
                attempted_rerank=True,
                attempted_sidecar=True,
                applied_rerank=False,
                applied_sidecar=False,
                succeeded=False,
                duration_ms=3.0,
            ),
        ]
    )

    payload = summary.to_payload()
    assert payload["completed"] is True
    assert payload["succeeded"] is False
    assert payload["duration_ms"] == 3.0
    assert payload["requested_rerank"] is True
    assert payload["requested_sidecar"] is True
    assert payload["attempted_rerank"] is True
    assert payload["attempted_sidecar"] is True
    assert payload["error_stages"] == ["search"]
    assert payload["error_types"] == ["RuntimeError"]


def test_summarize_search_trace_sanitizes_direct_event_labels() -> None:
    summary = summarize_search_trace(
        [
            SearchPlanned(
                channels=(_DENSE_PRIMARY_CHANNEL, "private channel token"),
                search_profile="private_profile_secret",
                fusion="rrf",
                plan_rerank="sk-secret-plan",
                metadata_filter="private_filter_secret",
                retrieve_stage="secret_stage_token_abc123",
                rerank_stage="ProviderRerank",
                postprocesses=("SidecarPostprocess", "private postprocess token"),
            ),
            SearchStageCompleted(
                stage="retrieve",
                stage_name="secret_stage_token_abc123",
            ),
            RerankApplied(
                provider="openai:sk-proj-abcdefghijklmnopqrstuvwxyz123456",
                model="anthropic:sk-ant-api03-abcdefghijklmnopqrstuvwxyz123456",
                fallback_reason="provider_token_timeout",
            ),
            SidecarApplied(
                provider="slack:xoxc-123456789012-123456789012-abcdefghijklmnopqrstuvwxyz",
                fallback_reason="sidecar_token_timeout",
            ),
            StageError(
                stage="private rerank stage",
                error_type="RuntimeError",
            ),
        ]
    )

    payload = summary.to_payload()
    rendered = repr(payload)
    assert "private_filter_secret" not in rendered
    assert "secret_stage_token_abc123" not in rendered
    assert "sk-proj-" not in rendered
    assert "sk-ant-api03-" not in rendered
    assert "xoxc-" not in rendered
    assert payload["channels"] == [_DENSE_PRIMARY_CHANNEL, "unknown"]
    assert payload["search_profile"] == "unknown"
    assert payload["plan_rerank"] == "unknown"
    assert payload["metadata_filter"] == "unknown"
    assert payload["retrieve_stage"] == "unknown"
    assert payload["rerank_stage"] == "ProviderRerank"
    assert payload["postprocesses"] == ["SidecarPostprocess", "unknown"]
    assert payload["rerank_provider"] == "unknown"
    assert payload["rerank_model"] == "unknown"
    assert payload["sidecar_provider"] == "unknown"
    assert payload["error_stages"] == ["unknown"]
    assert payload["error_types"] == ["RuntimeError"]


def test_summarize_search_trace_payloads_accepts_jsonl_event_shapes() -> None:
    summary = summarize_search_trace_payloads(
        [
            {
                "event_type": "search.started",
                "namespace": "tenant-secret",
                "corpus_ids": ["corpus-secret"],
                "query_length": 14,
                "limit": 3,
            },
            {
                "event_type": "search.planned",
                "namespace": "tenant-secret",
                "corpus_ids": ["corpus-secret"],
                "limit": 3,
                "final_limit": 3,
                "channels": [_DENSE_PRIMARY_CHANNEL],
                "prefetch_limits": [6],
                "search_profile": "fast",
                "fusion": "identity",
                "plan_rerank": TRACE_ABSENT_LABEL,
                "rerank_fallback_on_error": False,
            },
            {
                "event_type": "search.stage.completed",
                "stage": "context_pack",
                "stage_name": "build_context_pack",
                "candidate_count": 3,
                "result_count": 2,
                "dropped_count": 1,
                "duration_ms": 0.5,
            },
            {
                "event_type": "rerank.applied",
                "provider": "cohere",
                "model": "rerank-v3.5",
                "input_count": 6,
                "candidate_count": 3,
                "provider_result_count": 4,
                "accepted_count": 2,
                "dropped_count": 2,
                "rank_changed_count": 2,
                "rank_promoted_count": 1,
                "rank_demoted_count": 1,
                "max_rank_gain": 2,
                "max_rank_loss": 1,
                "provider_score_min": 0.32,
                "provider_score_max": 0.95,
                "search_score_min": 0.44,
                "search_score_max": 0.81,
                "result_count": 2,
                "top_k": 2,
                "fallback_reason": "TimeoutError",
                "truncation_reason": "candidate_count,max_output",
                "duration_ms": 1.75,
                "succeeded": False,
            },
            {
                "event_type": "sidecar.applied",
                "provider": "bm25",
                "input_count": 3,
                "provider_result_count": 5,
                "accepted_count": 4,
                "dropped_count": 1,
                "result_count": 4,
                "duration_ms": 0.25,
                "succeeded": True,
            },
            {
                "event_type": "search.completed",
                "namespace": "tenant-secret",
                "result_count": 2,
                "duration_ms": 4.0,
            },
        ]
    )

    payload = summary.to_payload()
    assert payload["completed"] is True
    assert payload["limit"] == 3
    assert payload["channels"] == [_DENSE_PRIMARY_CHANNEL]
    assert payload["search_profile"] == "fast"
    assert payload["rerank_fallback_on_error"] is False
    assert payload["attempted_rerank"] is True
    assert payload["applied_rerank"] is False
    assert payload["rerank_provider_result_count"] == 4
    assert payload["rerank_accepted_count"] == 2
    assert payload["rerank_dropped_count"] == 2
    assert payload["rerank_rank_changed_count"] == 2
    assert payload["rerank_rank_promoted_count"] == 1
    assert payload["rerank_rank_demoted_count"] == 1
    assert payload["rerank_max_rank_gain"] == 2
    assert payload["rerank_max_rank_loss"] == 1
    assert payload["rerank_provider_score_min"] == 0.32
    assert payload["rerank_provider_score_max"] == 0.95
    assert payload["rerank_search_score_min"] == 0.44
    assert payload["rerank_search_score_max"] == 0.81
    assert payload["rerank_fallback_reason"] == "TimeoutError"
    assert payload["rerank_truncation_reason"] == "candidate_count,max_output"
    assert payload["applied_sidecar"] is True
    assert payload["sidecar_provider_result_count"] == 5
    assert payload["sidecar_accepted_count"] == 4
    assert payload["sidecar_dropped_count"] == 1
    assert payload["stage_count"] == 1
    stages = cast(list[dict[str, object]], payload["stages"])
    assert stages[0]["stage"] == "context_pack"
    assert "tenant-secret" not in repr(payload)


def test_search_wrapper_error_not_double_counted_after_stage_error() -> None:
    summary = summarize_search_trace(
        [
            StageError(stage="rerank", error_type="TimeoutError"),
            StageError(stage="search", error_type="TimeoutError"),
        ]
    )

    payload = summary.to_payload()
    assert payload["error_count"] == 1
    assert payload["error_stages"] == ["rerank"]
    assert payload["error_types"] == ["TimeoutError"]


def test_search_wrapper_error_replaced_when_concrete_stage_arrives_later() -> None:
    summary = summarize_search_trace(
        [
            StageError(stage="search", error_type="TimeoutError"),
            StageError(stage="retrieve", error_type="TimeoutError"),
        ]
    )

    payload = summary.to_payload()
    assert payload["error_count"] == 1
    assert payload["error_stages"] == ["retrieve"]
    assert payload["error_types"] == ["TimeoutError"]


def test_summarize_search_trace_runs_uses_search_id_for_interleaved_events() -> None:
    summaries = summarize_search_trace_runs(
        [
            SearchStarted(search_id="search-a", query_length=4),
            SearchStarted(search_id="search-b", query_length=8),
            SearchStageCompleted(
                search_id="search-a",
                stage="retrieve",
                stage_name="HybridRetrieve",
                result_count=2,
            ),
            SearchCompleted(search_id="search-b", result_count=3),
            SearchCompleted(search_id="search-a", result_count=2),
        ]
    )

    payloads = [summary.to_payload() for summary in summaries]
    assert [payload["search_id"] for payload in payloads] == ["search-a", "search-b"]
    assert payloads[0]["query_length"] == 4
    assert payloads[0]["stage_count"] == 1
    assert payloads[0]["result_count"] == 2
    assert payloads[1]["query_length"] == 8
    assert payloads[1]["stage_count"] == 0
    assert payloads[1]["result_count"] == 3


def test_single_search_trace_payload_summary_rejects_multiple_search_ids() -> None:
    payloads = [
        {"event_type": "search.started", "search_id": "search-a"},
        {"event_type": "search.started", "search_id": "search-b"},
    ]

    try:
        summarize_search_trace_payloads(payloads)
    except ValueError as exc:
        assert str(exc) == "trace contains multiple search_ids"
    else:
        raise AssertionError("expected multiple search_ids to be rejected")


def test_trace_payload_runs_reject_unsafe_search_ids_before_grouping() -> None:
    payloads = [
        {"event_type": "search.started", "search_id": "tenant/a"},
        {"event_type": "search.started", "search_id": "tenant/b"},
    ]

    try:
        summarize_search_trace_payload_runs(payloads)
    except ValueError as exc:
        assert str(exc) == "trace field search_id must be a safe search identifier"
    else:
        raise AssertionError("expected unsafe search_id to be rejected")


def test_summarize_search_trace_runs_attaches_uncorrelated_stage_events_to_single_run() -> None:
    summaries = summarize_search_trace_runs(
        [
            SearchStarted(search_id="search-a", query_length=4, limit=2),
            SearchStageCompleted(
                stage="context_pack",
                stage_name="build_context_pack",
                result_count=1,
            ),
            SearchCompleted(search_id="search-a", result_count=1),
        ]
    )

    [summary] = summaries
    payload = summary.to_payload()
    assert payload["search_id"] == "search-a"
    assert payload["stage_count"] == 1
    stages = cast(list[dict[str, object]], payload["stages"])
    assert stages[0]["stage"] == "context_pack"


def test_summarize_search_trace_runs_rejects_ambiguous_uncorrelated_stage_events() -> None:
    try:
        summarize_search_trace_runs(
            [
                SearchStarted(search_id="search-a"),
                SearchStarted(search_id="search-b"),
                SearchStageCompleted(
                    stage="context_pack",
                    stage_name="build_context_pack",
                ),
            ]
        )
    except ValueError as exc:
        assert str(exc) == "trace contains mixed correlated and uncorrelated search events"
    else:
        raise AssertionError(
            "expected mixed correlated and uncorrelated search events to be rejected"
        )
