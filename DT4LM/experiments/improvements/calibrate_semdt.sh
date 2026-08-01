#!/usr/bin/env bash
# Run the judge backend selected by one complete experiment config.
set -euo pipefail

CONFIG_PATH="${1:?usage: calibrate_semdt.sh <experiment.yaml> [trajectory-run-dir]}"
TRAJECTORY_RUN_DIR="${2:-}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

COMMAND=(
  python
  "${PROJECT_ROOT}/experiments/improvements/calibrate_semdt.py"
  --config "${CONFIG_PATH}"
)

# A trajectory is optional because threshold calibration happens before SemDT.
if [[ -n "${TRAJECTORY_RUN_DIR}" ]]; then
  COMMAND+=(--trajectory-run-dir "${TRAJECTORY_RUN_DIR}")
fi
"${COMMAND[@]}"
