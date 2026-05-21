#!/usr/bin/env bash
# Materialize gitignored agent docs from docs/templates/. See scripts/README.md
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

copy_template() {
  local rel_src="$1"
  local rel_dest="$2"
  local src="${REPO_ROOT}/${rel_src}"
  local dest="${REPO_ROOT}/${rel_dest}"
  if [[ ! -f "$src" ]]; then
    echo "missing template: $rel_src" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$dest")"
  if [[ -f "$dest" && "$FORCE" -eq 0 ]]; then
    echo "skip (exists): $rel_dest"
    return 0
  fi
  cp "$src" "$dest"
  echo "wrote: $rel_dest"
}

cd "$REPO_ROOT"

copy_template "docs/templates/AGENTS.md" "AGENTS.md"
copy_template "docs/templates/AGENTS.md" "docs/AGENTS.md"
copy_template "docs/templates/CONTEXT.md" "docs/CONTEXT.md"
copy_template "docs/templates/MISSION.md" "MISSION.md"
copy_template "docs/templates/ROUTING.md" "docs/plans/ROUTING.md"

echo "agent docs ready (gitignored paths — do not git add)"
