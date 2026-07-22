#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${ACTIONSHIFT_PYTHON:-${PROJECT_DIR}/.venv/bin/python}"

cd "${PROJECT_DIR}"
"${PYTHON_BIN}" -m actionshift.evaluation.falsification \
  --output reports/week_one_gate.json \
  --card reports/benchmark_card.md \
  --episodes-per-seed 32 \
  --require-cuda
