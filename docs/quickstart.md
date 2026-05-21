# First 10 minutes

Prove `rag-core` is a real, inspectable retrieval engine — no managed-RAG account, no API keys, no hosted platform.

**Journey A** in [one-repo-retrieval-engine-strategy.md](plans/one-repo-retrieval-engine-strategy.md). After this path you should understand raw hits, model context, trace evidence, and a library eval gate.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- A clone of this repository (commands below assume repo root)

```bash
git clone https://github.com/kaanarici/rag-core.git
cd rag-core
uv sync
```

## One command (agents and CI)

```bash
./scripts/dx_smoke.sh
```

The script runs the same checks as the steps below and prints `dx smoke passed` on success.

### After `pip install` (no git checkout)

```bash
pip install rag-core
python -m rag_core.quickstart
```

Same demo as Step 5 below — context + citations from an installed wheel only.

## Step 1 — Shortest smoke (`demo`)

Confirms ingest, index, and search work with demo embeddings and in-memory Qdrant.

```bash
uv run rag-core demo --json
```

You should see JSON with `document_id`, `chunk_count`, and a non-empty `hits` array. Each hit includes `score`, `title`, and `text`.

## Step 2 — Folder search + raw hits (`local-search`)

Indexes `examples/demo_corpus` and runs a query. This is the main no-key proof on real files.

```bash
uv run rag-core local-search examples/demo_corpus \
  "How can invoices be paid?" \
  --events-jsonl /tmp/rag-core-events.jsonl \
  --json
```

You should see:

- `indexed_count` ≥ 1
- `hits` with at least one row containing invoice/payment language
- `corpus_id` and `namespace` reflecting the CLI defaults (`local` / derived corpus)

`local-search` returns **raw search hits**, not a model-ready context pack. Use Step 5 for context with citations.

## Step 3 — Trace evidence (inspectability)

Summarize the JSONL trace from Step 2. This is how you debug retrieval without a hosted vendor console.

```bash
uv run python - <<'PY'
import json
from pathlib import Path

from rag_core.events.traces import summarize_search_trace_payload_runs

path = Path("/tmp/rag-core-events.jsonl")
payloads = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
summaries = summarize_search_trace_payload_runs(payloads)
assert summaries, "expected at least one search trace run"
summary = summaries[-1]
assert summary.result_count > 0, summary
print(
    json.dumps(
        {
            "search_id": summary.search_id,
            "result_count": summary.result_count,
            "channels": list(summary.channels),
            "fusion": summary.fusion,
        },
        indent=2,
    )
)
PY
```

You should see `result_count` > 0 and channel/fusion fields describing the plan that ran.

## Step 4 — Runtime diagnostics (`doctor`)

Checks vector-store and embedding configuration shape without calling a paid provider.

```bash
uv run rag-core doctor \
  --qdrant-location :memory: \
  --embedding-provider demo \
  --embedding-dimensions 64 \
  --json
```

Inspect `vector_store` and `embedding` sections. Failures here usually mean conflicting Qdrant flags or missing extras.

## Step 5 — Model-ready context (`minimal_app`)

Shows `retrieve_context` with citations — the handoff surface your chat/agent code should use.

```bash
uv run python -m examples.minimal_app
```

You should see indexed chunk count, a `Context to pass into your model call` block, and citation lines.

## Step 6 — Retrieval quality gate (library eval)

Runs the slim `rag_core.evals` runner over a fixed local corpus (no `rag-core eval` CLI in v1).

```bash
uv run python -m examples.retrieval_eval
```

You should see a JSON report with per-case metrics and quality gates. Exit code `0` means the bundled thresholds passed.

## What this proves vs managed RAG

| You verified | Managed RAG usually hides |
|--------------|---------------------------|
| Parser/chunk/index/search path | Chunk boundaries and index updates |
| Raw hit JSON | Vendor-specific chunk schema only |
| Trace summary | Black-box retrieval plan |
| Context + citations | Opaque “context string” |
| Local eval gate | Hosted eval dashboards |

You did **not** need hosted connectors, auth, billing, or a retrieval SaaS account.

## Next steps

| Goal | Doc |
|------|-----|
| Embed in your application | [README — Library Usage](../README.md#library-usage) and `examples/minimal_app.py` |
| Self-host HTTP API | [self-host/quickstart.md](self-host/quickstart.md) |
| Contract reference | [expectations.md](expectations.md) |
| Product strategy | [plans/one-repo-retrieval-engine-strategy.md](plans/one-repo-retrieval-engine-strategy.md) |
