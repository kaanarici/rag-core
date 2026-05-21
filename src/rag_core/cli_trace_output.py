from __future__ import annotations

import json

from rag_core.events.traces import SearchTraceSummary


def emit_trace_summary(payload: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(_json_dumps(payload))
        return
    print(f"Completed: {payload.get('completed')}")
    print(
        "Query: "
        f"length={payload.get('query_length')} "
        f"limit={payload.get('limit')} "
        f"final_limit={payload.get('final_limit')} "
        f"corpora={payload.get('corpus_count')}"
    )
    print(
        "Plan: "
        f"channels={_format_trace_sequence(payload.get('channels'))} "
        f"fusion={payload.get('fusion') or 'none'} "
        f"rerank={payload.get('plan_rerank') or 'none'} "
        f"rerank_fallback={payload.get('rerank_fallback_on_error')} "
        f"sidecar={payload.get('use_sidecar')}"
    )
    print("Stages:")
    stages = payload.get("stages")
    if not isinstance(stages, list) or not stages:
        print("- none")
    else:
        for raw_stage in stages:
            stage = _require_mapping(raw_stage)
            print(
                f"- {stage.get('stage')}:{stage.get('stage_name')} "
                f"candidates={stage.get('candidate_count')} "
                f"results={stage.get('result_count')} "
                f"dropped={stage.get('dropped_count')} "
                f"truncated={stage.get('truncated')} "
                f"duration_ms={stage.get('duration_ms')}"
            )
    if payload.get("rerank_attempted") is True:
        print(_format_rerank_trace(payload))
    if payload.get("sidecar_attempted") is True:
        print(_format_sidecar_trace(payload))
    if isinstance(payload.get("embedding"), dict):
        print(_format_embedding_trace(_require_mapping(payload.get("embedding"))))
    if _intish(payload.get("error_count")):
        print(
            "Errors: "
            f"stages={_format_trace_sequence(payload.get('error_stages'))} "
            f"types={_format_trace_sequence(payload.get('error_types'))}"
        )


def emit_trace_summary_set(payload: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(_json_dumps(payload))
        return
    print(
        "Searches: "
        f"count={payload.get('search_count')} "
        f"completed={payload.get('completed_count')} "
        f"errors={payload.get('error_count')} "
        f"total_duration_ms={payload.get('total_duration_ms')}"
    )
    if _intish(payload.get("rerank_applied_count")) or _intish(
        payload.get("rerank_failed_count")
    ):
        print(
            "Rerank aggregate: "
            f"applied={payload.get('rerank_applied_count')} "
            f"failed={payload.get('rerank_failed_count')} "
            f"provider_results={payload.get('rerank_provider_result_count')} "
            f"accepted={payload.get('rerank_accepted_count')} "
            f"dropped={payload.get('rerank_dropped_count')} "
            f"rank_changed={payload.get('rerank_rank_changed_count')} "
            f"promoted={payload.get('rerank_rank_promoted_count')} "
            f"demoted={payload.get('rerank_rank_demoted_count')} "
            f"max_gain={payload.get('rerank_max_rank_gain')} "
            f"max_loss={payload.get('rerank_max_rank_loss')} "
            f"provider_score={_format_score_range(payload, 'rerank_provider_score')} "
            f"search_score={_format_score_range(payload, 'rerank_search_score')} "
            f"duration_ms={payload.get('rerank_duration_ms')}"
        )
    if _intish(payload.get("sidecar_applied_count")) or _intish(
        payload.get("sidecar_failed_count")
    ):
        print(
            "Sidecar aggregate: "
            f"applied={payload.get('sidecar_applied_count')} "
            f"failed={payload.get('sidecar_failed_count')} "
            f"provider_results={payload.get('sidecar_provider_result_count')} "
            f"accepted={payload.get('sidecar_accepted_count')} "
            f"dropped={payload.get('sidecar_dropped_count')} "
            f"duration_ms={payload.get('sidecar_duration_ms')}"
        )
    if isinstance(payload.get("embedding"), dict):
        print(_format_embedding_trace(_require_mapping(payload.get("embedding"))))
    searches = payload.get("searches")
    if not isinstance(searches, list):
        return
    for index, raw_search in enumerate(searches, start=1):
        search = _require_mapping(raw_search)
        print(
            f"- search {index}: "
            f"completed={search.get('completed')} "
            f"limit={search.get('limit')} "
            f"results={search.get('result_count')} "
            f"stages={search.get('stage_count')} "
            f"errors={search.get('error_count')} "
            f"duration_ms={search.get('duration_ms')}"
        )
        if search.get("rerank_attempted") is True:
            print(f"  {_format_rerank_trace(search)}")
        if search.get("sidecar_attempted") is True:
            print(f"  {_format_sidecar_trace(search)}")


def trace_summary_set_payload(
    summaries: tuple[SearchTraceSummary, ...],
) -> dict[str, object]:
    search_payloads = [summary.to_payload() for summary in summaries]
    return {
        "schema_version": 1,
        "search_count": len(search_payloads),
        "completed_count": sum(
            1 for payload in search_payloads if payload.get("completed") is True
        ),
        "error_count": sum(_intish(payload.get("error_count")) for payload in search_payloads),
        "total_duration_ms": sum(
            _floatish(payload.get("duration_ms")) for payload in search_payloads
        ),
        "rerank_applied_count": _count_search_true(
            search_payloads,
            "rerank_applied",
        ),
        "rerank_failed_count": _count_search_false(
            search_payloads,
            condition_key="rerank_attempted",
            value_key="rerank_succeeded",
        ),
        "rerank_provider_result_count": _sum_search_int(
            search_payloads,
            "rerank_provider_result_count",
        ),
        "rerank_accepted_count": _sum_search_int(
            search_payloads,
            "rerank_accepted_count",
        ),
        "rerank_dropped_count": _sum_search_int(
            search_payloads,
            "rerank_dropped_count",
        ),
        "rerank_rank_changed_count": _sum_search_int(
            search_payloads,
            "rerank_rank_changed_count",
        ),
        "rerank_rank_promoted_count": _sum_search_int(
            search_payloads,
            "rerank_rank_promoted_count",
        ),
        "rerank_rank_demoted_count": _sum_search_int(
            search_payloads,
            "rerank_rank_demoted_count",
        ),
        "rerank_max_rank_gain": _max_search_int(search_payloads, "rerank_max_rank_gain"),
        "rerank_max_rank_loss": _max_search_int(search_payloads, "rerank_max_rank_loss"),
        "rerank_provider_score_min": _min_search_float_when_accepted(
            search_payloads,
            "rerank_provider_score_min",
        ),
        "rerank_provider_score_max": _max_search_float_when_accepted(
            search_payloads,
            "rerank_provider_score_max",
        ),
        "rerank_search_score_min": _min_search_float_when_accepted(
            search_payloads,
            "rerank_search_score_min",
        ),
        "rerank_search_score_max": _max_search_float_when_accepted(
            search_payloads,
            "rerank_search_score_max",
        ),
        "rerank_duration_ms": _sum_search_float(search_payloads, "rerank_duration_ms"),
        "sidecar_applied_count": _count_search_true(
            search_payloads,
            "sidecar_applied",
        ),
        "sidecar_failed_count": _count_search_false(
            search_payloads,
            condition_key="sidecar_attempted",
            value_key="sidecar_succeeded",
        ),
        "sidecar_provider_result_count": _sum_search_int(
            search_payloads,
            "sidecar_provider_result_count",
        ),
        "sidecar_accepted_count": _sum_search_int(
            search_payloads,
            "sidecar_accepted_count",
        ),
        "sidecar_dropped_count": _sum_search_int(
            search_payloads,
            "sidecar_dropped_count",
        ),
        "sidecar_duration_ms": _sum_search_float(search_payloads, "sidecar_duration_ms"),
        "searches": search_payloads,
    }


def _format_trace_sequence(value: object) -> str:
    if not isinstance(value, list) or not value:
        return "none"
    return ",".join(str(item) for item in value)


def _format_rerank_trace(payload: dict[str, object]) -> str:
    return (
        "Rerank: "
        f"provider={payload.get('rerank_provider') or 'unknown'} "
        f"model={payload.get('rerank_model') or 'unknown'} "
        f"inputs={payload.get('rerank_input_count')} "
        f"candidates={payload.get('rerank_applied_candidate_count')} "
        f"provider_results={payload.get('rerank_provider_result_count')} "
        f"accepted={payload.get('rerank_accepted_count')} "
        f"dropped={payload.get('rerank_dropped_count')} "
        f"rank_changed={payload.get('rerank_rank_changed_count')} "
        f"promoted={payload.get('rerank_rank_promoted_count')} "
        f"demoted={payload.get('rerank_rank_demoted_count')} "
        f"max_gain={payload.get('rerank_max_rank_gain')} "
        f"max_loss={payload.get('rerank_max_rank_loss')} "
        f"provider_score={_format_score_range(payload, 'rerank_provider_score')} "
        f"search_score={_format_score_range(payload, 'rerank_search_score')} "
        f"results={payload.get('rerank_result_count')} "
        f"top_k={payload.get('rerank_top_k')} "
        f"fallback={payload.get('rerank_fallback_reason') or 'none'} "
        f"truncation={payload.get('rerank_truncation_reason') or 'none'} "
        f"succeeded={payload.get('rerank_succeeded')} "
        f"duration_ms={payload.get('rerank_duration_ms')}"
    )


def _format_sidecar_trace(payload: dict[str, object]) -> str:
    return (
        "Sidecar: "
        f"provider={payload.get('sidecar_provider') or 'unknown'} "
        f"inputs={payload.get('sidecar_input_count')} "
        f"provider_results={payload.get('sidecar_provider_result_count')} "
        f"accepted={payload.get('sidecar_accepted_count')} "
        f"dropped={payload.get('sidecar_dropped_count')} "
        f"results={payload.get('sidecar_result_count')} "
        f"fallback={payload.get('sidecar_fallback_reason') or 'none'} "
        f"succeeded={payload.get('sidecar_succeeded')} "
        f"duration_ms={payload.get('sidecar_duration_ms')}"
    )


def _format_embedding_trace(payload: dict[str, object]) -> str:
    return (
        "Embeddings: "
        f"requested_events={payload.get('requested_event_count')} "
        f"completed_events={payload.get('completed_event_count')} "
        f"requested_texts={payload.get('requested_text_count')} "
        f"completed_texts={payload.get('completed_text_count')} "
        f"dense_texts={payload.get('dense_completed_text_count')} "
        f"sparse_texts={payload.get('sparse_completed_text_count')} "
        f"cache_hits={payload.get('cache_hits')} "
        f"cache_misses={payload.get('cache_misses')} "
        f"cache_writes={payload.get('cache_writes')} "
        f"cache_bypasses={payload.get('cache_bypasses')} "
        f"duration_ms={payload.get('duration_ms')} "
        f"providers={_format_trace_sequence(payload.get('providers'))} "
        f"models={_format_trace_sequence(payload.get('models'))}"
    )


def _format_score_range(payload: dict[str, object], prefix: str) -> str:
    minimum = payload.get(f"{prefix}_min")
    maximum = payload.get(f"{prefix}_max")
    if not isinstance(minimum, int | float) or not isinstance(maximum, int | float):
        return "unknown"
    if isinstance(minimum, bool) or isinstance(maximum, bool):
        return "unknown"
    return f"{minimum}..{maximum}"


def _intish(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


def _floatish(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _float_score(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _count_search_true(searches: list[dict[str, object]], key: str) -> int:
    return sum(1 for search in searches if search.get(key) is True)


def _count_search_false(
    searches: list[dict[str, object]],
    *,
    condition_key: str,
    value_key: str,
) -> int:
    return sum(
        1
        for search in searches
        if search.get(condition_key) is True and search.get(value_key) is False
    )


def _sum_search_int(searches: list[dict[str, object]], key: str) -> int:
    return sum(_intish(search.get(key)) for search in searches)


def _max_search_int(searches: list[dict[str, object]], key: str) -> int:
    return max((_intish(search.get(key)) for search in searches), default=0)


def _sum_search_float(searches: list[dict[str, object]], key: str) -> float:
    return sum(_floatish(search.get(key)) for search in searches)


def _min_search_float_when_accepted(
    searches: list[dict[str, object]],
    key: str,
) -> float | None:
    values = [
        value
        for search in searches
        if _intish(search.get("rerank_accepted_count"))
        if (value := _float_score(search.get(key))) is not None
    ]
    return min(values, default=None)


def _max_search_float_when_accepted(
    searches: list[dict[str, object]],
    key: str,
) -> float | None:
    values = [
        value
        for search in searches
        if _intish(search.get("rerank_accepted_count"))
        if (value := _float_score(search.get(key))) is not None
    ]
    return max(values, default=None)


def _require_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _json_dumps(payload: dict[str, object]) -> str:
    return json.dumps(payload, allow_nan=False, indent=2, sort_keys=True)
