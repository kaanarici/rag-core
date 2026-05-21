# Retrieval Quality Evals

`rag-core` evals measure the retrieval layer before your app calls a model. Use them for CI gates, retrieval-profile comparisons, reranker decisions, and regression checks after parser, chunking, embedding, or vector-store changes.

They do not evaluate generated answers. Keep answer-quality tests in your app, where prompts, tools, auth scope, and model policy live.

## Case Format

Eval cases are JSONL. Each non-empty line is one labelled query:

```json
{"case_id": "help/billing-payment-methods", "query": "How can customers pay invoices?", "namespace": "acme", "corpus_ids": ["help-center"], "expected_chunk_ids": ["billing.md"]}
```

Fields:

- `case_id`: optional stable identifier for CI output and long-lived reports. When set, it must be unique in the JSONL file.
- `query`: the user query to run through `core.search(...)`.
- `namespace`: the application-owned namespace to search.
- `corpus_ids`: one or more corpus partitions to search.
- `expected_chunk_ids`: relevant `SearchResult.id` values. The runner falls back to `document_id` when fixtures are document-level.
- `expected_grades`: optional relevance grades for nDCG@10. Omit it for binary relevance.

In a source checkout, [`examples/eval_cases.jsonl`](../../examples/eval_cases.jsonl) is a small fixture that matches `examples/demo_corpus`. The `examples/` directory is not installed into the wheel. Installed-package users should keep eval cases in their own app repo and pass that path to the CLI or `load_cases(...)`.

Source-checkout smoke test that creates a small demo corpus, runs three eval cases, attaches run metadata, and enforces quality gates:

```bash
uv run python -m examples.retrieval_eval
```

## Library runner (v1)

v1 ships evals as a **library** and checkout example. There is no `rag-core eval` CLI.

Run the checkout smoke that indexes `examples/demo_corpus`, evaluates three cases, applies quality gates, and prints a redacted JSON report:

```bash
uv run python -m examples.retrieval_eval
```

Build your own worker or CI script with `load_cases`, `run_eval`, `eval_report`, `add_quality_gate`, and `eval_exit_code`. Attach a `run` metadata object so reports record vector store, embedding model, search profile, and rerank settings.

`eval_report(...)` returns per-case metrics (`recall_at_5`, `recall_at_10`, `mrr`, `ndcg_at_10`, `latency_ms`). Call `redact_eval_report(...)` before publishing artifacts to shared logs. Quality gates return exit code `1` when a configured floor or ceiling fails.

For retrieval debugging, pair eval runs with `search --events-jsonl` and `summarize_search_trace(...)` from `rag_core.events` on the written JSONL file. There is no `trace-summary` CLI in v1.

## Library Usage

Use the same runner and report builders inside a worker, CI script, or service:

```python
from pathlib import Path

from rag_core import RAGCore
from rag_core.evals import eval_report, load_cases, run_eval


async def evaluate(core: RAGCore) -> dict[str, object]:
    cases = load_cases(Path("cases.jsonl"))
    results = await run_eval(core, cases, rerank=False)
    return eval_report(results)
```

Attach thresholds with `add_quality_gate(report, metrics, thresholds)` and exit with `eval_exit_code(report)` for CI gates.

In a source checkout, [`examples/retrieval_eval.py`](../../examples/retrieval_eval.py) shows the same library path as a runnable script with no external service or API key.

## Operating Rules

- Keep fixtures safe to commit. Queries and expected ids should not contain secrets, private URLs, or customer text.
- Prefer `case_id` over raw query text when triaging CI output.
- Raise floors as retrieval improves. Do not lower a floor without recording why in the change that lowers it.
- Keep fast synthetic evals in unit/CI paths; run larger benchmark corpora as explicit opt-in jobs.
- Evaluate retrieval changes before broad provider, parser, reranker, or search-profile changes.
