# First 10 minutes

Two modes: **Smoke** (no API keys, deterministic embeddings) and **Configured** (your
Qdrant + embedding provider). Smoke proves the pipeline; configured proves your stack.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- A local checkout

```bash
uv sync
```

## Smoke (no API keys)

### One command (CI and local smoke)

```bash
./scripts/dx_smoke.sh
```

### After package install

```bash
uv pip install -e .
python -m rag_core.quickstart
```

Wheel installs run the same module; `scripts/wheel_smoke.py` verifies this from a
fresh consumer venv after `uv build`.

### Step 1 — Shortest smoke (`demo`)

```bash
uv run rag-core demo --json
```

Expect `document_id`, `chunk_count`, and non-empty `hits`. Demo embeddings are not
semantic search.

### Step 2 — Folder search (`local-search`)

```bash
uv run rag-core local-search examples/demo_corpus \
  "How can invoices be paid?" \
  --events-jsonl /tmp/rag-core-events.jsonl \
  --json
```

Expect `indexed_count` ≥ 1 and payment-related hits. Raw hits only — use Step 5 for
context packs.

Prefer `--json` for scripts. Without it, `local-search` prints an indexed/skipped
breakdown and a truncation hint for real folders.

### Step 3 — Trace evidence

```bash
uv run python - <<'PY'
import json
from pathlib import Path
from rag_core.events.traces import summarize_search_trace_payload_runs

path = Path("/tmp/rag-core-events.jsonl")
payloads = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
summaries = summarize_search_trace_payload_runs(payloads)
summary = summaries[-1]
assert summary.result_count > 0, summary
print(json.dumps({"search_id": summary.search_id, "result_count": summary.result_count, "channels": list(summary.channels), "fusion": summary.fusion}, indent=2))
PY
```

### Step 4 — `doctor`

```bash
uv run rag-core doctor \
  --qdrant-location :memory: \
  --embedding-provider demo \
  --embedding-dimensions 64 \
  --json
```

Drop `--json` when you want human next-step hints. Default doctor output points
unconfigured first runs toward either the no-key `local-search` smoke or the
provider/store flags needed for configured retrieval.

### Step 5 — Prompt-safe context

```bash
uv run python -m examples.minimal_app
```

### Step 6 — Folder eval (`local-eval`)

```bash
uv run rag-core local-eval examples/demo_corpus examples/eval_cases.jsonl \
  --min-recall-at-5 1 \
  --min-mrr 1 \
  --json
```

Expect `case_count`, aggregate metrics, and a passing `quality_gate`. The command
uses the `namespace` and `corpus_ids` from the JSONL cases, then indexes local
documents and resolves relative-path `expected_ids` such as `billing.md` to the
indexed local document keys.

## Configured (your stack)

Requires embedding API credentials (example uses OpenAI) and Qdrant:

```bash
docker compose up -d qdrant
export OPENAI_API_KEY=sk-...
uv run rag-core ingest examples/demo_corpus --namespace acme --corpus-id help \
  --qdrant-url http://127.0.0.1:6333 \
  --embedding-provider openai --embedding-model text-embedding-3-small --embedding-dimensions 1536
uv run rag-core retrieve-context "How can invoices be paid?" \
  --namespace acme --corpus-id help \
  --qdrant-url http://127.0.0.1:6333 \
  --embedding-provider openai --embedding-model text-embedding-3-small --embedding-dimensions 1536
```

Add `--rerank` when a reranker extra is installed and configured. See [embed.md](embed.md).

## Library eval (your cases)

```bash
uv run python -m examples.retrieval_eval
```

Uses `rag_core.evals` in the repo — not the CI fixture corpus. Cases use
`expected_ids` for relevant chunk or document ids and can add prompt-context
assertions:

```json
{"query": "billing policy", "namespace": "acme", "corpus_ids": ["help"], "expected_ids": ["billing.md"], "expected_context_contains": ["ACH"], "forbidden_context_contains": ["content_sha256", "document_key"]}
```

Exit code `0` means bundled example thresholds passed.

## Next steps

| Goal | Where |
|------|--------|
| Embed in your app | [embed.md](embed.md), `examples/embedded_service.py` |
| Self-host HTTP | [self-host.md](self-host.md) |
| Contracts | [expectations.md](expectations.md) |
| Providers | [providers.md](providers.md) |
