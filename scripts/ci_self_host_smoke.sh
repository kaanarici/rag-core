#!/usr/bin/env bash
# CI wrapper for Journey C: start serve, wait for readiness, run HTTP smoke.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PORT="${PORT:-8787}"
BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT}}"

uv run rag-core serve \
  --host 127.0.0.1 \
  --port "$PORT" \
  --qdrant-location :memory: \
  --embedding-provider demo \
  --embedding-dimensions 64 &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" >/dev/null 2>&1 || true
  wait "$SERVER_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for _ in $(seq 1 60); do
  if curl -sf "$BASE_URL/health/ready" >/dev/null; then
    BASE_URL="$BASE_URL" ./scripts/self_host_smoke.sh
    exit 0
  fi
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "rag-core serve exited before readiness" >&2
    wait "$SERVER_PID"
  fi
  sleep 0.5
done

echo "rag-core serve did not become ready" >&2
exit 1
