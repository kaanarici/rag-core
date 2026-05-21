#!/usr/bin/env bash
# Ensure README/compose display name matches dev/project_identity.toml.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DISPLAY_NAME="$(python3 - <<'PY'
from pathlib import Path
import tomllib
root = Path("dev")
paths = [root / "project_identity.local.toml", root / "project_identity.toml"]
for path in paths:
    if path.is_file():
        with path.open("rb") as fh:
            data = tomllib.load(fh)
        print(data["display"]["name"])
        break
PY
)"

if ! head -1 README.md | grep -q "# ${DISPLAY_NAME}"; then
  echo "README title does not match display.name=${DISPLAY_NAME}" >&2
  exit 1
fi

echo "brand check ok (${DISPLAY_NAME})"
