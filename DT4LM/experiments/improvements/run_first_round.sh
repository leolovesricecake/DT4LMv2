#!/usr/bin/env bash
# Run exactly one experiment so failures and retries stay independent.
set -euo pipefail

DATASET_CONFIG="${1:?usage: run_first_round.sh <dataset-config.yaml> <experiment-config.yaml> [runner options]}"
EXPERIMENT_CONFIG="${2:?usage: run_first_round.sh <dataset-config.yaml> <experiment-config.yaml> [runner options]}"
shift 2
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python "${PROJECT_ROOT}/experiments/improvements/run_improvements.py" \
  --dataset-config "${DATASET_CONFIG}" \
  --experiment-config "${EXPERIMENT_CONFIG}" \
  "$@"
