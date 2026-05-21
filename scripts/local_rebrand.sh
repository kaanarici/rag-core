#!/usr/bin/env bash
# Replace display name in human-facing files only (never src/rag_core or tests).
# Usage: ./scripts/local_rebrand.sh NEW_DISPLAY_NAME
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 NEW_DISPLAY_NAME" >&2
  echo "example: $0 atlas-retrieval" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

IDENTITY="$REPO_ROOT/dev/project_identity.toml"
LOCAL_IDENTITY="$REPO_ROOT/dev/project_identity.local.toml"
if [[ -f "$LOCAL_IDENTITY" ]]; then
  IDENTITY="$LOCAL_IDENTITY"
fi

OLD_NAME="$(python3 - <<'PY' "$IDENTITY"
import sys
import tomllib
with open(sys.argv[1], "rb") as fh:
    data = tomllib.load(fh)
print(data["display"]["name"])
PY
)"
NEW_NAME="$1"

if [[ "$OLD_NAME" == "$NEW_NAME" ]]; then
  echo "display name already $NEW_NAME"
  exit 0
fi

echo "replacing display name: $OLD_NAME -> $NEW_NAME"
echo "(product package rag_core / CLI rag-core unchanged)"

replace_in() {
  local file="$1"
  if [[ -f "$file" ]]; then
    sed -i '' "s/${OLD_NAME}/${NEW_NAME}/g" "$file"
  fi
}

for file in README.md docs/quickstart.md docs/self-host/quickstart.md compose.yaml; do
  replace_in "$file"
done

sed -i '' "s/^name = .*/name = \"${NEW_NAME}\"/" "$IDENTITY"

echo "done. review git diff; run ./scripts/brand_check.sh"
