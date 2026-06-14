#!/usr/bin/env bash
# CI wrapper for Journey C: start serve, wait for readiness, run HTTP smoke.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${PORT:-}" ]]; then
  PORT="$(python - <<'PY'
import socket

with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)"
fi
BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT}}"
RUNTIME_TMPDIR="$(mktemp -d)"
JOB_DB_PATH="${JOB_DB_PATH:-$RUNTIME_TMPDIR/jobs.sqlite3}"
SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
  rm -rf "$RUNTIME_TMPDIR"
}
trap cleanup EXIT

uv run rag-core serve \
  --host 127.0.0.1 \
  --port "$PORT" \
  --job-db-path "$JOB_DB_PATH" \
  --qdrant-location :memory: \
  --embedding-provider demo \
  --embedding-model demo-dense-v1 \
  --embedding-dimensions 64 &
SERVER_PID=$!

for _ in $(seq 1 60); do
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "rag-core serve exited before readiness" >&2
    wait "$SERVER_PID"
  fi
  if curl -sf "$BASE_URL/health/ready" >/dev/null; then
    if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
      echo "rag-core serve exited after readiness probe" >&2
      wait "$SERVER_PID"
    fi
    BASE_URL="$BASE_URL" ./scripts/self_host_smoke.sh
    exit 0
  fi
  sleep 0.5
done

echo "rag-core serve did not become ready" >&2
exit 1
