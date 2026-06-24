#!/usr/bin/env bash
set -euo pipefail

# Validate checked-in provider_contract JSON fixtures (no network, no API keys).
echo "Validating provider contract fixtures"
uv run pytest -m provider_contract -q "$@"

echo "Review tests/fixtures/providers/ for Authorization or api-key leakage before commit."
