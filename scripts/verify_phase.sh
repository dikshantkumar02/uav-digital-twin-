#!/usr/bin/env bash
# Run the PHASE 1 + PHASE 2 validation + test suite.
# Exits non-zero if anything fails.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo ">>> Validating configuration"
python -m backend.config.cli --config config --validate-only

echo
echo ">>> Running PHASE 1 + PHASE 2 tests"
python -m pytest backend/tests/ -v

echo
echo ">>> All phases verified: OK"
