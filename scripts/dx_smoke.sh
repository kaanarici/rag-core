#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv run rag-core demo --json
uv run rag-core search "How can invoices be paid?" examples/demo_corpus --demo --json
uv run rag-core context "How can invoices be paid?" examples/demo_corpus --demo --json
uv run rag-core doctor --json
uv run python -m rag_core.quickstart
uv run python -m examples.retrieval_eval
