#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:---full}" in
  --quick) marker='not live and not eval and not integration and not provider_contract' ;;
  --full) marker='not live' ;;
  *) echo 'Usage: ./scripts/landing_check.sh [--quick|--full]' >&2; exit 2 ;;
esac
uv sync --group dev
uv run ruff check .
uv run mypy src tests examples
uv run pytest -q -m "$marker"
./scripts/dx_smoke.sh
if [[ "${1:---full}" == --full ]]; then
  uv build
  uv run python scripts/check_dist_artifacts.py
  uv run python scripts/wheel_smoke.py
fi
