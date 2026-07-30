#!/usr/bin/env bash
# Run or resume exactly one judge backend and optional trajectory audit.
set -euo pipefail

DATASET_CONFIG="${1:?usage: calibrate_semdt.sh <dataset-config.yaml> <judge-config.secert.yaml> [trajectory-run-dir]}"
JUDGE_CONFIG="${2:?usage: calibrate_semdt.sh <dataset-config.yaml> <judge-config.secert.yaml> [trajectory-run-dir]}"
TRAJECTORY_RUN_DIR="${3:-}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

COMMAND=(
  python
  "${PROJECT_ROOT}/experiments/improvements/calibrate_semdt.py"
  --dataset-config "${DATASET_CONFIG}"
  --judge-config "${JUDGE_CONFIG}"
)

# A trajectory is optional because threshold calibration happens before SemDT.
if [[ -n "${TRAJECTORY_RUN_DIR}" ]]; then
  COMMAND+=(--trajectory-run-dir "${TRAJECTORY_RUN_DIR}")
fi
"${COMMAND[@]}"
