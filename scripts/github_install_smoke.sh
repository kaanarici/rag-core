#!/usr/bin/env bash
# Install rag-core from a Git remote/ref in a fresh venv and run installed smoke.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/github_install_smoke.sh [remote-url] [ref]

Defaults:
  remote-url  git config --get remote.origin.url
  ref         current checkout HEAD SHA
EOF
}

case "${1:-}" in
  "-h" | "--help")
    usage
    exit 0
    ;;
esac

if (($# > 2)); then
  usage >&2
  exit 2
fi

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
remote_url="${1:-$(git -C "$repo_root" config --get remote.origin.url)}"
ref="${2:-$(git -C "$repo_root" rev-parse HEAD)}"

if [[ -z "$remote_url" ]]; then
  echo "No remote URL supplied and origin is not configured." >&2
  exit 2
fi

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/rag-core-git-install.XXXXXX")"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

python_bin="$tmp_dir/venv/bin/python"
cli_bin="$tmp_dir/venv/bin/rag-core"
requirement="rag-core @ git+${remote_url}@${ref}"

if [[ "$remote_url" == git+* ]]; then
  requirement="rag-core @ ${remote_url}@${ref}"
fi

echo "==> Creating fresh venv"
uv venv "$tmp_dir/venv" >/dev/null

echo "==> Installing $requirement"
uv pip install --python "$python_bin" "$requirement" >/dev/null

echo "==> Running installed quickstart"
"$python_bin" -m rag_core.quickstart >/dev/null

echo "==> Running installed CLI demo"
"$cli_bin" demo --json >"$tmp_dir/demo.json"
"$python_bin" - "$tmp_dir/demo.json" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not payload.get("hits"):
    raise SystemExit("installed CLI demo returned no hits")
PY

echo "Git install smoke passed"
