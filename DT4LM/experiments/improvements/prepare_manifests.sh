#!/usr/bin/env bash
# Freeze train/test manifests from exactly one tracked dataset configuration.
set -euo pipefail

CONFIG_PATH="${1:?usage: prepare_manifests.sh <dataset-config.yaml>}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python "${PROJECT_ROOT}/statistics/prepare_manifests.py" \
  --config "${CONFIG_PATH}"
