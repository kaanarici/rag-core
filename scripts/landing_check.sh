#!/usr/bin/env bash
# Local v0 beta landing gate; see scripts/README.md.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

FAST_PYTEST_MARKER='not live and not eval and not eval_harness and not provider_contract and not integration'

usage() {
  cat <<'EOF'
Usage: ./scripts/landing_check.sh [--quick|--full]

  --quick  Run the local iteration gate: sync, lint, typecheck, fast pytest, dx smoke.
  --full   Run the full v0 beta landing gate (default).
EOF
}

mode="full"
case "${1:-}" in
  "")
    ;;
  "--quick")
    mode="quick"
    ;;
  "--full")
    ;;
  "-h" | "--help")
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

if (($# > 1)); then
  usage >&2
  exit 2
fi

run_step() {
  echo "==> $*"
  "$@"
}

run_step uv sync --group dev
run_step uv run ruff check .
run_step uv run mypy src tests examples
if [[ "$mode" == "quick" ]]; then
  run_step uv run pytest -q -m "$FAST_PYTEST_MARKER"
  run_step ./scripts/dx_smoke.sh
  echo "quick landing check passed"
  exit 0
fi

run_step uv run pytest -q
run_step ./scripts/dx_smoke.sh
run_step ./scripts/verify_vercel_ai_sdk_example.sh
run_step ./scripts/ci_self_host_smoke.sh
run_step uv build
run_step uv run python scripts/check_dist_artifacts.py
run_step uv run python scripts/wheel_smoke.py

echo "landing check passed"
