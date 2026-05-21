from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_core.cli_trace_output import (
    emit_trace_summary,
    emit_trace_summary_set,
    trace_summary_set_payload,
)
from rag_core.events.embedding_trace_summary import summarize_embedding_trace_payloads
from rag_core.events.traces import (
    summarize_search_trace_payload_runs,
    summarize_search_trace_payloads,
)


def run_trace_summary(args: argparse.Namespace) -> int:
    path = Path(args.path)
    payloads: list[dict[str, object]] = []
    try:
        lines = _load_jsonl_trace_payloads(path)
    except OSError:
        raise ValueError("trace file: unable to read trace file") from None
    for line_number, payload in lines:
        try:
            summarize_search_trace_payloads([payload])
            summarize_embedding_trace_payloads([payload])
        except ValueError as exc:
            raise ValueError(f"trace file: line {line_number}: {exc}") from None
        payloads.append(payload)
    embedding_summary = summarize_embedding_trace_payloads(payloads)
    summaries = summarize_search_trace_payload_runs(payloads)
    if not summaries and embedding_summary.has_events:
        payload = trace_summary_set_payload(())
        payload["embedding"] = embedding_summary.to_payload()
        emit_trace_summary_set(payload, as_json=args.json)
        return 0
    if len(summaries) > 1:
        payload = trace_summary_set_payload(summaries)
        if embedding_summary.has_events:
            payload["embedding"] = embedding_summary.to_payload()
        emit_trace_summary_set(payload, as_json=args.json)
        return 0
    summary = summaries[0] if summaries else summarize_search_trace_payloads(payloads)
    payload = summary.to_payload()
    if embedding_summary.has_events:
        payload["embedding"] = embedding_summary.to_payload()
    emit_trace_summary(payload, as_json=args.json)
    return 0


def _load_jsonl_trace_payloads(path: Path) -> list[tuple[int, dict[str, object]]]:
    payloads: list[tuple[int, dict[str, object]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                parsed = json.loads(
                    raw_line,
                    object_pairs_hook=_reject_duplicate_trace_keys,
                    parse_constant=_reject_json_constant,
                )
            except json.JSONDecodeError:
                raise ValueError(f"trace file: line {line_number}: invalid JSON") from None
            except ValueError as exc:
                raise ValueError(f"trace file: line {line_number}: {exc}") from None
            if not isinstance(parsed, dict):
                raise ValueError(f"trace file: line {line_number}: expected JSON object")
            payloads.append(
                (line_number, {str(key): value for key, value in parsed.items()})
            )
    return payloads


def _reject_duplicate_trace_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _reject_json_constant(value: str) -> None:
    _ = value
    raise ValueError("invalid JSON constant")
