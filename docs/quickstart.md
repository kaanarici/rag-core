# First 10 minutes

Prove `rag-core` is a real, inspectable retrieval engine — no managed-RAG account, no API keys.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Repo root

```bash
git clone https://github.com/kaanarici/rag-core.git
cd rag-core
uv sync
```

## One command (agents and CI)

```bash
./scripts/dx_smoke.sh
```

### After `pip install` (no git checkout)

```bash
pip install rag-core
python -m rag_core.quickstart
```

## Step 1 — Shortest smoke (`demo`)

```bash
uv run rag-core demo --json
```

Expect `document_id`, `chunk_count`, and non-empty `hits`.

## Step 2 — Folder search (`local-search`)

```bash
uv run rag-core local-search examples/demo_corpus \
  "How can invoices be paid?" \
  --events-jsonl /tmp/rag-core-events.jsonl \
  --json
```

Expect `indexed_count` ≥ 1 and payment-related hits. Raw hits only — use Step 5 for context packs.

## Step 3 — Trace evidence

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

## Step 4 — `doctor`

```bash
uv run rag-core doctor \
  --qdrant-location :memory: \
  --embedding-provider demo \
  --embedding-dimensions 64 \
  --json
```

## Step 5 — Model context

```bash
uv run python -m examples.minimal_app
```

## Step 6 — Library eval

```bash
uv run python -m examples.retrieval_eval
```

Uses `rag_core.evals` — there is no `rag-core eval` CLI. Exit code `0` means bundled thresholds passed.

## Next steps

| Goal | Where |
|------|--------|
| Embed in your app | [README](../README.md#embed-in-your-app), `examples/embedded_service.py` |
| Self-host HTTP | [self-host.md](self-host.md) |
| Contracts | [expectations.md](expectations.md) |
| Providers | [providers.md](providers.md) |
