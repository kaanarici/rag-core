# First 10 minutes

Two modes: **Smoke** (no API keys, deterministic embeddings) and **Configured** (your
Qdrant + embedding provider). Smoke proves the pipeline; configured proves your stack.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Repo root

```bash
uv sync
```

## Smoke (no API keys)

### One command (agents and CI)

```bash
./scripts/dx_smoke.sh
```

### After editable install

```bash
uv pip install -e .
python -m rag_core.quickstart
```

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

### Step 5 — Model context

```bash
uv run python -m examples.minimal_app
```

## Configured (your stack)

Requires embedding API credentials (example uses OpenAI) and Qdrant:

```bash
docker compose up -d qdrant
export OPENAI_API_KEY=sk-...
uv run rag-core ingest examples/demo_corpus --namespace acme --corpus-id help \
  --qdrant-url http://127.0.0.1:6333 \
  --embedding-provider openai --embedding-model text-embedding-3-small --embedding-dimensions 1536
uv run rag-core retrieve-context "How can invoices be paid?" \
  --namespace acme --corpus-id help --json
```

Add `--rerank` when a reranker extra is installed and configured. See [embed.md](embed.md).

## Library eval (your cases)

```bash
uv run python -m examples.retrieval_eval
```

Uses `rag_core.evals` in the repo — not the CI fixture corpus. Exit code `0` means
bundled example thresholds passed.

## Next steps

| Goal | Where |
|------|--------|
| Embed in your app | [embed.md](embed.md), `examples/embedded_service.py` |
| Self-host HTTP | [self-host.md](self-host.md) |
| Contracts | [expectations.md](expectations.md) |
| Providers | [providers.md](providers.md) |
