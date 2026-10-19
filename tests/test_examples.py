"""Examples must keep running against the public surface."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from examples.chatbot_context import run_chatbot_context_demo


def _module_env() -> dict[str, str]:
    env = os.environ.copy()
    root = Path.cwd()
    pythonpath = [str(root / "src"), str(root)]
    if existing := env.get("PYTHONPATH"):
        pythonpath.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return env


def test_rag_core_quickstart_runs_as_module() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "rag_core.quickstart"],
        check=True,
        cwd=Path.cwd(),
        env=_module_env(),
        capture_output=True,
        text=True,
    )
    assert "Prompt-safe context text:" in result.stdout
    assert "Citations:" in result.stdout


def test_minimal_app_runs_as_module() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "examples.minimal_app"],
        check=True,
        cwd=Path.cwd(),
        env=_module_env(),
        capture_output=True,
        text=True,
    )

    assert "Prompt-safe context text:" in result.stdout
    assert "Citations:" in result.stdout
    assert "billing.txt" in result.stdout


def test_ask_folder_example_is_copyable_local_path() -> None:
    source = Path("examples/ask_folder.py").read_text(encoding="utf-8")
    nonblank = [line for line in source.splitlines() if line.strip()]

    assert len(nonblank) <= 12
    assert "from rag_core.easy import index" in source
    assert "with index(" in source
    assert ".context(" in source
    assert "Config" not in source
    assert ".add(" not in source
    assert "rag_core.demo" not in source
    assert "ChunkingConfig" not in source


def test_easy_index_empty_folder_raises_clean_error(tmp_path: Path) -> None:
    from rag_core.easy import index

    with pytest.raises(RuntimeError) as excinfo:
        index(tmp_path, persist_dir=tmp_path.parent / "indexes")
    message = str(excinfo.value)
    assert "indexed no files" in message
    # The day-one facade must not leak lower-level glob/ingest guidance.
    assert "glob" not in message
    assert "supported files matched" not in message


@pytest.mark.integration
def test_ask_folder_example_runs_with_easy_index_path() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.ask_folder",
            "examples/demo_corpus",
            "How can invoices be paid?",
        ],
        cwd=Path.cwd(),
        env=_module_env(),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Citations:" in result.stdout
    assert "invoice" in result.stdout.lower() or "pay" in result.stdout.lower()


def test_configured_retrieval_requires_explicit_credentials() -> None:
    env = _module_env()
    env.pop("OPENAI_API_KEY", None)
    result = subprocess.run(
        [sys.executable, "-m", "examples.configured_retrieval"],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "OPENAI_API_KEY is required" in result.stderr
    assert not result.stdout


def test_prompt_context_examples_use_prompt_safe_citation_summary() -> None:
    for path in (
        Path("examples/chatbot_context.py"),
        Path("examples/configured_retrieval.py"),
        Path("examples/minimal_app.py"),
        Path("src/rag_core/quickstart.py"),
    ):
        source = path.read_text(encoding="utf-8")
        assert ".prompt_citation_summary" in source
        assert ".citations" not in source
        assert ".source_id" not in source
        assert ".result_id" not in source
        assert ".document_key" not in source


def test_python_examples_import_config_and_engine_from_their_owners() -> None:
    for path in Path("examples").glob("*.py"):
        source = path.read_text(encoding="utf-8")

        assert "from rag_core.core_models import Config" not in source
        assert "from rag_core import Engine" not in source
        assert "from rag_core.ingest.remote_keys import" not in source


def test_chatbot_context_example_returns_context_pack() -> None:
    context = asyncio.run(run_chatbot_context_demo("How do I pay an invoice?"))

    assert context.snippets
    assert context.snippets[0].citation_id
    assert "billing" in context.as_prompt_text().lower()


def test_retrieval_eval_module_prints_redacted_report() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "examples.retrieval_eval"],
        cwd=Path.cwd(),
        env=_module_env(),
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 0
    assert payload["run"]["purpose"] == "wiring_only_demo_embeddings"
    assert payload["run"]["retrieval_mode"] == "hybrid"
    assert payload["cases"][0]["case_label"] == "case-1"
    assert "case_id" not in payload["cases"][0]
    assert "query" not in payload["cases"][0]
    assert "expected_ids" not in payload["cases"][0]
    assert "expected_chunk_ids" not in payload["cases"][0]
    assert "How can a customer update payment details?" not in result.stdout


def test_configured_eval_help_smoke_without_model_download() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "examples.configured_eval", "--help"],
        cwd=Path.cwd(),
        env=_module_env(),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "corpus_dir" in result.stdout
    assert "--provider {local,openai}" in result.stdout


@pytest.mark.integration
def test_configured_eval_runs_with_local_provider() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.configured_eval",
            "examples/demo_corpus",
            "examples/eval_cases.jsonl",
        ],
        cwd=Path.cwd(),
        env=_module_env(),
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 0, result.stderr
    assert payload["run"]["mode"] == "local"
    assert payload["run"]["embedding_model"] == "BAAI/bge-small-en-v1.5"
    assert payload["metrics"]["recall_at_5"] == 1.0
    assert payload["metrics"]["mrr"] == 1.0
