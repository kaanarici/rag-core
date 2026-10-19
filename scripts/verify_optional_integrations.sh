#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --group dev --extra mcp
uv run pytest -q tests/test_mcp_server.py tests/test_mcp_cli.py
