#!/usr/bin/env bash
# Type-check the copyable Vercel AI SDK example against current AI SDK v6 declarations.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP_ROOT="$(mktemp -d)"

cleanup() {
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

cp "$REPO_ROOT/examples/vercel_ai_sdk_search_tool.ts" "$TMP_ROOT/vercel_ai_sdk_search_tool.ts"

cat >"$TMP_ROOT/package.json" <<'JSON'
{
  "private": true,
  "type": "module",
  "devDependencies": {
    "@types/json-schema": "latest",
    "@types/node": "latest",
    "ai": "^6.0.0",
    "typescript": "latest"
  }
}
JSON

cat >"$TMP_ROOT/tsconfig.json" <<'JSON'
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "strict": true,
    "skipLibCheck": false,
    "lib": ["ES2022", "DOM"],
    "types": ["node"]
  },
  "include": ["vercel_ai_sdk_search_tool.ts"]
}
JSON

(
  cd "$TMP_ROOT"
  npm install --silent --no-audit --no-fund
  node -e 'const pkg = require("./node_modules/ai/package.json"); console.log(`ai@${pkg.version}`);'
  ./node_modules/.bin/tsc --noEmit --pretty false
)

echo "vercel AI SDK example typecheck passed"
