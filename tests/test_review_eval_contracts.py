"""Eval review regressions: local Qdrant lifecycle and safety plumbing."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import cast

import pytest

from rag_core.cli.commands.eval import run_local_eval_command
from rag_core.cli.parser import _build_parser
from rag_core.core import Config, Engine
from rag_core.demo import DemoEmbeddingProvider
from rag_core.evals import EvalCase, eval_report, run_eval
from rag_core.events.sink import EventSink
from rag_core.ingest.sources.local import document_key
from rag_core.search.vector_models import SearchResult
from tests.support import make_search_result


@pytest.fixture
def clean_eval_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in os.environ:
        if key.startswith("RAG_CORE_"):
            monkeypatch.delenv(key)


def _deterministic_engine(config: Config, *, event_sink: EventSink | None) -> Engine:
    return Engine(
        config, embedding_provider=DemoEmbeddingProvider(), event_sink=event_sink
    )


@pytest.mark.integration
@pytest.mark.parametrize("scan", ["complete", "truncated", "incomplete"])
def test_persistent_cli_eval_prunes_only_after_complete_scan(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    clean_eval_env: None,
    monkeypatch: pytest.MonkeyPatch,
    scan: str,
) -> None:
    folder = tmp_path / "corpus"
    folder.mkdir()
    old = folder / "old.txt"
    old.write_text("Old invoices policy.")
    (folder / "current-a.txt").write_text("Current invoices policy.")
    (folder / "current-b.txt").write_text("Additional invoices policy.")
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        json.dumps(
            {
                "query": "invoices",
                "tenant_id": "t",
                "collections": ["c"],
                "expected_ids": [document_key(folder, old)],
            }
        )
        + "\n"
    )
    argv = [
        "eval",
        str(folder),
        str(cases),
        "--persist-dir",
        str(tmp_path / "index"),
        "--mode",
        "dense",
        "--embedding-provider",
        "demo",
        "--embedding-dimensions",
        "64",
        "--json",
    ]

    def evaluate(extra: list[str]) -> dict[str, object]:
        args = _build_parser().parse_args([*argv, *extra])
        assert (
            asyncio.run(
                run_local_eval_command(
                    args, event_sink=None, core_factory=_deterministic_engine
                )
            )
            == 0
        )
        return cast(dict[str, object], json.loads(capsys.readouterr().out))

    initial = evaluate([])
    assert cast(dict[str, object], initial["metrics"])["recall_at_5"] == 1.0
    old.rename(tmp_path / "retained-old.txt")
    if scan == "incomplete":
        original_stat = Path.stat
        stat_calls = 0

        def failing_stat(path: Path, *, follow_symlinks: bool = True) -> os.stat_result:
            nonlocal stat_calls
            if path == folder / "current-a.txt" and not follow_symlinks:
                stat_calls += 1
                raise OSError("injected discovery failure")
            return original_stat(path, follow_symlinks=follow_symlinks)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "stat", failing_stat)
            with pytest.raises(OSError, match="injected discovery failure"):
                evaluate([])
        assert stat_calls >= 1
        retained = evaluate(["--max-files", "1"])
        assert cast(dict[str, object], retained["metrics"])["recall_at_5"] == 1.0
    truncated = scan == "truncated"
    report = evaluate(["--max-files", "1"] if truncated else [])
    run = cast(dict[str, object], report["run"])
    metrics = cast(dict[str, object], report["metrics"])
    assert run["truncated"] is truncated
    assert run["indexed_count"] == (1 if truncated else 2)
    assert metrics["recall_at_5"] == (1.0 if truncated else 0.0)


@pytest.mark.integration
@pytest.mark.parametrize("file_input", [False, True])
def test_cli_eval_does_not_request_pruning_without_manifest_or_folder(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    clean_eval_env: None,
    file_input: bool,
) -> None:
    folder = tmp_path / "corpus"
    folder.mkdir()
    source = folder / "billing.txt"
    source.write_text("Invoices use card or ACH.")
    path = source if file_input else folder
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        json.dumps(
            {
                "query": "invoices",
                "tenant_id": "t",
                "collections": ["c"],
                "expected_ids": ["billing.txt"],
            }
        )
        + "\n"
    )
    args = _build_parser().parse_args(
        [
            "eval",
            str(path),
            str(cases),
            "--qdrant-location",
            ":memory:",
            "--mode",
            "dense",
            "--embedding-provider",
            "demo",
            "--json",
            *(["--manifest-dir", str(tmp_path / "manifest")] if file_input else []),
        ]
    )
    assert (
        asyncio.run(
            run_local_eval_command(
                args, event_sink=None, core_factory=_deterministic_engine
            )
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["run"]["indexed_count"] == 1
    assert report["metrics"]["recall_at_5"] == 1.0


@pytest.mark.plumbing
@pytest.mark.parametrize(
    "forbidden",
    [
        ("chunk-1",),
        ("private-document",),
        ("private-key",),
        ("chunk-1", "private-document", "private-key"),
    ],
)
def test_forbidden_hit_identities_are_independent_of_zero_grade_judgment(
    forbidden: tuple[str, ...],
) -> None:
    hit = make_search_result(
        id="chunk-1", document_id="private-document", document_key="private-key"
    )

    class HitCore:
        async def search(self, **_: object) -> list[SearchResult]:
            return [hit]

    case = EvalCase(
        query="example",
        tenant_id="t",
        collections=["c"],
        expected_ids=["good-document"],
        expected_grades={"good-document": 3, "chunk-1": 0},
        forbidden_ids=forbidden,
    )
    [result] = asyncio.run(run_eval(cast(Engine, HitCore()), [case]))
    assert result.retrieved_ids == ("chunk-1",)
    assert result.recall_at_5 == 0.0
    assert result.forbidden_id_count == 1
    assert result.prompt_safety_pass is False


@pytest.mark.plumbing
@pytest.mark.parametrize("max_concurrency", [1, 2])
def test_failed_cases_fail_safety_checks_and_remain_in_pass_rate_denominator(
    max_concurrency: int,
) -> None:
    class MixedCore:
        async def search(self, **kwargs: object) -> list[SearchResult]:
            if kwargs["query"] != "good":
                raise RuntimeError("private provider failure")
            return [make_search_result(document_id="doc", text="mandatory answer")]

    def case(query: str, *, no_results: bool = False) -> EvalCase:
        return EvalCase(
            query=query,
            tenant_id="t",
            collections=["c"],
            expected_ids=[] if no_results else ["doc"],
            expect_no_results=no_results,
            expected_context_contains=["mandatory"],
            forbidden_context_contains=["forbidden"],
            forbidden_private_identifiers=["private-path"],
            forbidden_ids=["private-document"],
            expected_citation_count_min=1,
            expected_source_count_min=1,
        )

    results = asyncio.run(
        run_eval(
            cast(Engine, MixedCore()),
            [case("good"), case("bad"), case("bad-empty", no_results=True)],
            max_concurrency=max_concurrency,
        )
    )
    for result in results[1:]:
        assert result.error_type == "RuntimeError"
        assert result.context_contains_pass is False
        assert result.prompt_safety_pass is False
        assert result.no_result_pass is False
        assert result.context_recall == 0.0
    report = eval_report(results)
    assert report["failure_count"] == 2
    metrics = cast(dict[str, object], report["metrics"])
    assert metrics["prompt_safety_pass_rate"] == pytest.approx(1 / 3)
    assert metrics["context_recall"] == pytest.approx(1 / 3)
    assert metrics["no_result_pass_rate"] == 0.0
    for payload in cast(list[dict[str, object]], report["cases"])[1:]:
        assert payload["ok"] is False
        assert payload["context_contains_pass"] is False
        assert payload["prompt_safety_pass"] is False
    failed_metrics = cast(dict[str, object], eval_report(results[1:])["metrics"])
    assert failed_metrics["prompt_safety_pass_rate"] == 0.0
