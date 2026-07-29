#!/usr/bin/env bash
# Journey A smoke. See scripts/README.md
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

TRACE_FILE="${TRACE_FILE:-/tmp/rag-core-dx-smoke-trace.jsonl}"
rm -f "$TRACE_FILE"

echo "step 1: demo"
demo_json="$(uv run rag-core demo --json)"
python3 -c '
import json, sys
payload = json.loads(sys.stdin.read())
assert payload.get("chunk_count", 0) > 0, payload
assert payload.get("hits"), payload
' <<<"$demo_json"
echo "demo: ok"

echo "step 2: search"
search_json="$(
  uv run rag-core search \
    "How can invoices be paid?" \
    examples/demo_corpus \
    --demo \
    --trace-jsonl "$TRACE_FILE" \
    --json
)"
python3 -c '
import json, sys
payload = json.loads(sys.stdin.read())
hits = payload.get("hits")
assert isinstance(hits, list) and hits, payload
text = " ".join(str(hit.get("text", "")) for hit in hits).lower()
assert "invoice" in text or "pay" in text or "billing" in text, text[:200]
' <<<"$search_json"
echo "search: ok"

echo "step 3: trace summary"
TRACE_FILE="$TRACE_FILE" uv run python - <<'PY'
import json
import os
from pathlib import Path

from rag_core.events.traces import summarize_search_trace_payload_runs

path = Path(os.environ["TRACE_FILE"])
payloads = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
summaries = summarize_search_trace_payload_runs(payloads)
assert summaries, "expected search trace summaries"
summary = summaries[-1]
assert summary.result_count > 0, summary
print(f"trace: ok (result_count={summary.result_count})")
PY

echo "step 4: doctor"
uv run rag-core doctor \
  --qdrant-location :memory: \
  --embedding-provider demo \
  --embedding-dimensions 64 \
  --json >/dev/null
echo "doctor: ok"

echo "step 5: context retrieval (minimal_app)"
minimal_out="$(
  uv run python -m examples.minimal_app 2>&1
)"
if ! grep -q "Prompt-safe context text:" <<<"$minimal_out"; then
  echo "$minimal_out" >&2
  echo "minimal_app missing context output" >&2
  exit 1
fi
echo "context retrieval: ok"

echo "step 6: retrieval eval"
uv run python -m examples.retrieval_eval >/dev/null
echo "retrieval eval: ok"

echo "step 7: eval"
eval_json="$(
  uv run rag-core eval examples/demo_corpus examples/eval_cases.jsonl --json
)"
python3 -c '
import json, sys
payload = json.loads(sys.stdin.read())
assert payload.get("case_count") == 3, payload
run = payload.get("run") or {}
assert run.get("indexed_count") == 3, payload
metrics = payload.get("metrics") or {}
assert metrics.get("recall_at_5") == 1.0, payload
assert metrics.get("mrr") == 1.0, payload
' <<<"$eval_json"
echo "eval: ok"

echo "dx smoke passed"
