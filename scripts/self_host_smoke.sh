#!/usr/bin/env bash
# Journey C smoke. See scripts/README.md
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8787}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INGEST_PATH="${INGEST_PATH:-$REPO_ROOT/examples/demo_corpus/billing.md}"

curl -sf "$BASE_URL/health" >/dev/null
echo "health: ok"

curl -sf "$BASE_URL/health/ready" >/dev/null
echo "health/ready: ok"

curl -sf "$BASE_URL/v1/runtime" >/dev/null
echo "runtime: ok"

ingest="$(
  curl -sf -X POST "$BASE_URL/v1/ingest" \
    -H 'Content-Type: application/json' \
    -d "$(python3 -c 'import json,sys; print(json.dumps({"path": sys.argv[1], "collection": "help"}))' "$INGEST_PATH")"
)"
python3 -c 'import json,sys; p=json.load(sys.stdin); assert p.get("document_id") and p.get("chunk_count", 0) > 0, p' <<<"$ingest"
echo "ingest: ok"

hits="$(
  curl -sf -X POST "$BASE_URL/v1/search" \
    -H 'Content-Type: application/json' \
    -d '{"query":"How can invoices be paid?","collection":"help","limit":3}'
)"
python3 -c 'import json,sys; hits=json.load(sys.stdin); assert isinstance(hits,list) and hits, hits' <<<"$hits"
echo "search: ok (${#hits} bytes)"

curl -sf -X POST "$BASE_URL/v1/search/context" \
  -H 'Content-Type: application/json' \
  -d '{"query":"invoice payment","collection":"help","limit":3}' \
  | python3 -c 'import json,sys; p=json.load(sys.stdin); assert p.get("context_text")' >/dev/null
echo "context retrieval: ok"

echo "self-host smoke passed"
