#!/usr/bin/env bash
# Clone the public checkout into a temp directory and run release-relevant proof.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/public_checkout_smoke.sh [--quick|--package] [remote-url] [ref]

  --quick    Clone and run ./scripts/landing_check.sh --quick.
  --package  Clone, run quick gate, build, artifact check, and wheel smoke.

Defaults:
  remote-url  git config --get remote.origin.url
  ref         current checkout HEAD SHA
EOF
}

mode="package"
case "${1:-}" in
  "--quick")
    mode="quick"
    shift
    ;;
  "--package" | "")
    [[ "${1:-}" == "--package" ]] && shift
    ;;
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

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/rag-core-public-checkout.XXXXXX")"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

echo "==> Cloning $remote_url"
git clone --quiet "$remote_url" "$tmp_dir/repo"
cd "$tmp_dir/repo"

echo "==> Checking out $ref"
git checkout --quiet "$ref"

echo "==> Running quick landing gate"
./scripts/landing_check.sh --quick

if [[ "$mode" == "quick" ]]; then
  echo "public checkout quick smoke passed"
  exit 0
fi

echo "==> Building artifacts"
uv build

echo "==> Checking artifacts"
uv run python scripts/check_dist_artifacts.py

echo "==> Running wheel smoke"
uv run python scripts/wheel_smoke.py

echo "public checkout package smoke passed"
