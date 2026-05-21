#!/usr/bin/env bash
# Smoke-test a running rag-core serve instance (Path A/B from docs/self-host/quickstart.md).
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

job_id="$(
  curl -sf -X POST "$BASE_URL/v1/ingest" \
    -H 'Content-Type: application/json' \
    -d "$(python3 -c "import json,sys; print(json.dumps({'path': sys.argv[1], 'namespace': 'acme', 'corpus_id': 'help'}))" "$INGEST_PATH")" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])'
)"
echo "ingest job: $job_id"

for _ in $(seq 1 60); do
  status="$(curl -sf "$BASE_URL/v1/ingest/$job_id" | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')"
  if [[ "$status" == "completed" ]]; then
    echo "ingest: completed"
    break
  fi
  if [[ "$status" == "failed" ]]; then
    curl -sf "$BASE_URL/v1/ingest/$job_id"
    echo >&2
    echo "ingest job failed" >&2
    exit 1
  fi
  sleep 0.2
done

if [[ "${status:-}" != "completed" ]]; then
  echo "ingest job timed out" >&2
  exit 1
fi

hits="$(
  curl -sf -X POST "$BASE_URL/v1/search" \
    -H 'Content-Type: application/json' \
    -d '{"query":"How can invoices be paid?","namespace":"acme","corpus_ids":["help"],"limit":3}'
)"
python3 -c 'import json,sys; hits=json.load(sys.stdin); assert isinstance(hits,list) and hits, hits' <<<"$hits"
echo "search: ok (${#hits} bytes)"

curl -sf -X POST "$BASE_URL/v1/retrieve-context" \
  -H 'Content-Type: application/json' \
  -d '{"query":"invoice payment","namespace":"acme","corpus_ids":["help"],"limit":3}' \
  | python3 -c 'import json,sys; p=json.load(sys.stdin); assert p.get("context_text")' >/dev/null
echo "retrieve-context: ok"

echo "self-host smoke passed"
