#!/usr/bin/env bash

set -euo pipefail

if (( $# < 2 )); then
  echo "Usage: $0 <dataset> <cuda_devices> [model_id]" >&2
  exit 2
fi

DATASET="$1"
CUDA_DEVICES="$2"
MODEL_ID="${3:-albertbasev1-v2}"  # albertbasev1-v2  gpt1-2  debertabasev1-v3

METHODS=(
  dt4lm-kuleshov
  dt4lm-fastga
  dt4lm-leap
  ff-pbs
  dynamic-beam
  ff-pareto-greedy
  hard-pbs
  ff-mnew
  ffms-greedy
  hard-ffms
)

FAILED_METHODS=()

for method in "${METHODS[@]}"; do
  config_path="experiments/improvements/configs/${DATASET}/${MODEL_ID}-${method}.yaml"

  echo
  echo "============================================================"
  echo "Running method: ${method}"
  echo "Config: ${config_path}"
  echo "============================================================"

  if CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
    bash experiments/improvements/run_first_round.sh "${config_path}"; then
    echo "[SUCCESS] ${method}"
  else
    exit_code=$?
    echo "[FAILED] ${method} (exit code: ${exit_code})" >&2
    FAILED_METHODS+=("${method}")
  fi
done

echo
echo "============================================================"
echo "Execution summary"
echo "============================================================"

if (( ${#FAILED_METHODS[@]} == 0 )); then
  echo "All methods completed successfully."
  exit 0
fi

echo "${DATASET}|${MODEL_ID} Failed methods (${#FAILED_METHODS[@]}):" >&2
printf '  - %s\n' "${FAILED_METHODS[@]}" >&2

# 所有方法均已尝试执行，但只要存在失败项，脚本最终返回非零状态。
exit 1