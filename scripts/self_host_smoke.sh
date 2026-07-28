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

job_id="$(
  curl -sf -X POST "$BASE_URL/v1/ingest" \
    -H 'Content-Type: application/json' \
    -d "$(python3 -c 'import json,sys; print(json.dumps({"path": sys.argv[1], "collection": "help"}))' "$INGEST_PATH")" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])'
)"
echo "ingest job: $job_id"

events="$(curl -sfN --max-time 60 "$BASE_URL/v1/ingest/$job_id/events")"
python3 -c 'import json,sys; statuses=[json.loads(line[6:])["status"] for line in sys.stdin if line.startswith("data: ")]; assert statuses and statuses[-1]=="completed", statuses' <<<"$events"
echo "ingest events: ok"

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
