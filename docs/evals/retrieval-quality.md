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

## CLI

Run a baseline retrieval eval against an already indexed collection:

```bash
uv run rag-core eval \
  --cases cases.jsonl \
  --qdrant-url http://localhost:6333 \
  --search-profile balanced \
  --min-recall-at-5 0.8 \
  --min-mrr 0.7 \
  --max-mean-latency-ms 250 \
  --max-p95-latency-ms 500 \
  --min-throughput-qps 4 \
  --events-jsonl traces/eval.jsonl \
  --json
```

Compare common search profiles before changing defaults:

```bash
uv run rag-core eval \
  --cases cases.jsonl \
  --qdrant-url http://localhost:6333 \
  --compare-search-profiles balanced fast lexical \
  --min-mrr 0.7 \
  --json
```

Profile comparisons use a deterministic baseline profile: the lexicographically smallest profile name. Library callers can set a different baseline explicitly. Reordering `--compare-search-profiles` arguments does not change deltas.

Compare reranking against the same cases:

```bash
uv run rag-core eval \
  --cases cases.jsonl \
  --qdrant-url http://localhost:6333 \
  --compare-rerank \
  --reranker-provider cohere \
  --reranker-model rerank-english-v3.0 \
  --min-ndcg-at-10 0.8 \
  --json
```

Quality gates return exit code `1` when a configured floor or ceiling fails. Latency gates include mean latency, p95 latency, and throughput so a fast average cannot hide a slow tail. In rerank comparison mode, gates apply to the reranked result. In search-profile comparison mode, gates apply to every compared profile.

CLI JSON output includes a top-level `run` object for single evals and per-branch `run` objects inside rerank and profile comparisons. By default, `rag-core eval --json` redacts raw case/query identifiers and emits stable case ordinals, labels, counts, metrics, quality gates, and run metadata.

Use `rag-core eval --json --json-raw` only when you need raw `case_id`, `query`, `namespace`, `corpus_ids`, `expected_chunk_ids`, `expected_grades`, and `retrieved_ids` for debugging. Treat raw report artifacts as sensitive when cases contain customer text, private corpus names, tenant scopes, or internal document IDs.

Library helpers such as `eval_report(...)` and `eval_comparison_report(...)` return the raw report shape. Call `redact_eval_report(...)` before publishing those payloads to shared logs or CI artifacts. The `examples/retrieval_eval.py` script prints a redacted report while still using the raw report for the quality-gate exit code.

Run metadata records the runtime and retrieval shape under test: `rag-core` version, Python version, vector store, physical collection or namespace, embedding model, search profile or query-plan preset, rerank toggle, reranker provider, and rerank budget. Reports do not include API keys or provider secrets.

Add `--events-jsonl` when an eval failure needs the same sanitized search trace events that `search --events-jsonl` writes. The trace file can be summarized with `rag-core trace-summary traces/eval.jsonl --json`; event payloads preserve query length, plan shape, stages, embedding cache totals, applied rerank and sidecar diagnostics, and timing without writing raw query text. Rerank trace summaries include accepted and dropped provider results, aggregate rank movement, and accepted provider or search score ranges. Multi-case eval traces are reported as multiple search summaries so one slow or failed case does not get collapsed into the final case; multi-search summaries also include aggregate applied rerank and sidecar counters.

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

For profile or rerank comparisons, build reports with `eval_profile_comparison_report(...)` or `eval_comparison_report(...)`, then attach thresholds with `add_quality_gate(...)`.

In a source checkout, [`examples/retrieval_eval.py`](../../examples/retrieval_eval.py) shows the same library path as a runnable script with no external service or API key.

## Operating Rules

- Keep fixtures safe to commit. Queries and expected ids should not contain secrets, private URLs, or customer text.
- Prefer `case_id` over raw query text when triaging CI output.
- Raise floors as retrieval improves. Do not lower a floor without recording why in the change that lowers it.
- Keep fast synthetic evals in unit/CI paths; run larger benchmark corpora as explicit opt-in jobs.
- Evaluate retrieval changes before broad provider, parser, reranker, or search-profile changes.
