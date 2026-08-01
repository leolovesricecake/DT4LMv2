#!/usr/bin/env bash
# Run exactly one complete experiment so failures and retries stay independent.
set -euo pipefail

CONFIG_PATH="${1:?usage: run_first_round.sh <experiment.yaml>}"
shift
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python "${PROJECT_ROOT}/experiments/improvements/run_improvements.py" \
  --config "${CONFIG_PATH}" \
  "$@"
