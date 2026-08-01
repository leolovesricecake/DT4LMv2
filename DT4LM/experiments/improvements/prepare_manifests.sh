#!/usr/bin/env bash
# Freeze test and optional calibration samples from one complete experiment.
set -euo pipefail

CONFIG_PATH="${1:?usage: prepare_manifests.sh <experiment.yaml>}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python "${PROJECT_ROOT}/statistics/prepare_manifests.py" \
  --config "${CONFIG_PATH}"
