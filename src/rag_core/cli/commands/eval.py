from __future__ import annotations
from rag_core.search.request_models import SearchMode

import argparse
import json
from collections.abc import Mapping, Sequence, Callable
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

from rag_core.cli.config import config_from_args
from rag_core.core import Engine
from rag_core.demo import build_demo_core
from rag_core.evals import (
    EvalCase,
    eval_report,
    load_cases,
    redact_eval_report,
    run_eval,
)
from rag_core.evals.report_models import EvalReport
from rag_core.ingest.local import LocalIngestResult, LocalIngestSuccess
from rag_core.ingest.sources.local import reject_local_symlink_path

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink


async def run_local_eval_command(
    args: argparse.Namespace,
    *,
    event_sink: "EventSink | None",
    core_factory: Callable[..., Engine] = Engine,
) -> int:
    path = _validated_local_path(args.path)
    cases = load_cases(Path(args.cases))
    tenant_id, collection = _eval_scope(cases)
    max_concurrency = _positive_int(args.max_concurrency, flag="--max-concurrency")
    config = None if args.demo else config_from_args(args, path=path)
    mode: SearchMode = args.mode or (
        "hybrid" if config is not None and config.hybrid else "dense"
    )
    core = (
        build_demo_core(store_collection="local_eval", event_sink=event_sink)
        if args.demo
        else core_factory(config, event_sink=event_sink)
    )
    async with core:
        payload = await _index_and_eval(
            core,
            path=path,
            cases=cases,
            tenant_id=tenant_id,
            collection=collection,
            max_files=args.max_files,
            max_concurrency=max_concurrency,
            mode=mode,
            prune_missing=path.is_dir()
            and config is not None
            and config.ingest.manifest_directory is not None,
        )
    redacted = redact_eval_report(payload)
    _emit_local_eval(redacted, as_json=args.json)
    return _report_exit_code(payload)


async def _index_and_eval(
    core: Engine,
    *,
    path: Path,
    cases: Sequence[EvalCase],
    tenant_id: str,
    collection: str,
    max_files: int,
    max_concurrency: int,
    mode: SearchMode,
    prune_missing: bool = False,
) -> EvalReport:
    ingest = await core.add(
        path,
        tenant_id=tenant_id,
        collection=collection,
        max_files=max_files,
        max_concurrency=max_concurrency,
        prune_missing=prune_missing,
    )
    if ingest.succeeded_count == 0:
        raise ValueError("no files could be indexed under %s" % path)
    eval_cases = _resolve_eval_case_ids(cases, ingest)
    eval_started = perf_counter()
    results = await run_eval(
        core,
        eval_cases,
        mode=mode,
        max_concurrency=max_concurrency,
    )
    return eval_report(
        results,
        run=_eval_run_metadata(
            core=core,
            mode=mode,
            max_concurrency=max_concurrency,
            wall_clock_seconds=perf_counter() - eval_started,
            tenant_id=tenant_id,
            collection=collection,
            ingest=ingest,
        ),
    )


def _eval_scope(cases: Sequence[EvalCase]) -> tuple[str, str]:
    tenant_ids = {case.tenant_id for case in cases}
    if len(tenant_ids) != 1:
        raise ValueError("eval cases must share one tenant_id")
    collections = {case.collections for case in cases}
    if len(collections) != 1:
        raise ValueError("eval cases must share one collections value")
    only_collections = next(iter(collections))
    if len(only_collections) != 1:
        raise ValueError("eval cases must target exactly one collection")
    return next(iter(tenant_ids)), only_collections[0]


def _resolve_eval_case_ids(
    cases: Sequence[EvalCase],
    ingest: LocalIngestResult,
) -> list[EvalCase]:
    document_key_by_alias = _document_key_aliases(ingest.succeeded)
    return [
        EvalCase(
            query=case.query,
            tenant_id=case.tenant_id,
            collections=case.collections,
            expected_ids=tuple(
                document_key_by_alias.get(expected_id, expected_id)
                for expected_id in case.expected_ids
            ),
            expected_grades=_resolved_expected_grades(
                case.expected_grades,
                document_key_by_alias,
            ),
            case_id=case.case_id,
            expected_context_contains=case.expected_context_contains,
            forbidden_context_contains=case.forbidden_context_contains,
            forbidden_private_identifiers=case.forbidden_private_identifiers,
            expected_citation_count_min=case.expected_citation_count_min,
            expected_source_count_min=case.expected_source_count_min,
            max_context_chars=case.max_context_chars,
            max_context_tokens=case.max_context_tokens,
            document_ids=case.document_ids,
            forbidden_ids=tuple(
                document_key_by_alias.get(forbidden_id, forbidden_id)
                for forbidden_id in case.forbidden_ids
            ),
            expect_no_results=case.expect_no_results,
        )
        for case in cases
    ]


def _document_key_aliases(
    records: Sequence[LocalIngestSuccess],
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for record in records:
        aliases[record.document_key] = record.document_key
        local_alias = _local_document_key_alias(record.document_key)
        if local_alias is not None:
            aliases.setdefault(local_alias, record.document_key)
    return aliases


def _local_document_key_alias(document_key: str) -> str | None:
    if not document_key.startswith("local:"):
        return None
    public_key, separator, _source = document_key.removeprefix("local:").partition(
        "#source:"
    )
    if not separator or not public_key:
        return None
    return public_key


def _resolved_expected_grades(
    grades: Mapping[str, int] | None,
    document_key_by_alias: Mapping[str, str],
) -> Mapping[str, int] | None:
    if grades is None:
        return None
    return {
        document_key_by_alias.get(expected_id, expected_id): grade
        for expected_id, grade in grades.items()
    }


def _eval_run_metadata(
    *,
    core: Engine,
    mode: SearchMode,
    max_concurrency: int,
    wall_clock_seconds: float,
    tenant_id: str,
    collection: str,
    ingest: LocalIngestResult,
) -> EvalReport:
    runtime = core.describe_runtime()
    embedding = _mapping(runtime.get("embedding"))
    return {
        "mode": "local_eval",
        "vector_store": _mapping(runtime.get("vector_store")).get("provider"),
        "embedding_model": embedding.get("model"),
        "embedding_note": (
            "deterministic demo embeddings check wiring, not semantic retrieval quality"
            if embedding.get("model") == "demo-dense-v1"
            else "retrieval quality depends on your corpus and labelled cases"
        ),
        "rerank": False,
        "max_concurrency": max_concurrency,
        "wall_clock_seconds": wall_clock_seconds,
        "tenant_id": tenant_id,
        "collection": collection,
        "indexed_count": ingest.succeeded_count,
        "skipped_count": ingest.failed_count,
        "skipped_failed_count": ingest.failed_count,
        "truncated": ingest.truncated,
        "retrieval_mode": mode,
    }


def _validated_local_path(raw: str) -> Path:
    path = Path(raw)
    if not path.exists():
        raise FileNotFoundError(str(path))
    reject_local_symlink_path(path)
    return path


def _emit_local_eval(payload: EvalReport, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    run = _mapping(payload.get("run"))
    metrics = _mapping(payload.get("metrics"))
    print(f"Indexed: {_payload_int(run, 'indexed_count')} files")
    print(f"Failed: {_payload_int(run, 'skipped_count')} files")
    if run.get("truncated") is True:
        print("Truncated: yes; rerun with --max-files to include more supported files")
    print(f"Corpus: {run.get('tenant_id')}/{run.get('collection')}")
    embedding_model = run.get("embedding_model")
    embedding_note = run.get("embedding_note")
    if isinstance(embedding_model, str) and isinstance(embedding_note, str):
        print(f"Embeddings: {embedding_model} ({embedding_note})")
    print(f"Cases: {_payload_int(payload, 'case_count')}")
    print(
        "Metrics: "
        f"recall@5={_payload_float(metrics, 'recall_at_5'):.3f} "
        f"mrr={_payload_float(metrics, 'mrr'):.3f} "
        f"latency_p95_ms={_payload_float(metrics, 'latency_p95_ms'):.1f}"
    )


def _positive_int(value: int, *, flag: str) -> int:
    if value <= 0:
        raise ValueError(f"{flag} must be positive")
    return value


def _report_exit_code(report: EvalReport) -> int:
    return (
        1
        if _payload_int(report, "failure_count") > 0
        or _payload_int(_mapping(report.get("run")), "skipped_count") > 0
        else 0
    )


def _mapping(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    return {}


def _payload_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return 0


def _payload_float(payload: dict[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return 0.0


__all__ = ["run_local_eval_command"]
