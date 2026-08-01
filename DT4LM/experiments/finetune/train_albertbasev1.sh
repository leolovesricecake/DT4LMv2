#!/usr/bin/env bash
# Keep every fine-tuning hyperparameter symmetric with the ALBERT-v2 run.
set -euo pipefail

DATASET="$1"
CUDA_DEVICES="$2"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DATASET_PATH="${PROJECT_ROOT}/outputs/datasets/${DATASET}"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/finetuned/albertbasev1_${DATASET}"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" textattack train \
  --model-name-or-path /mnt/huawei/nsq/models/albert/albert-base-v1 \
  --dataset "${DATASET_PATH}" \
  --model-max-length 256 \
  --per-device-train-batch-size 64 \
  --per-device-eval-batch-size 256 \
  --epochs 8 \
  --learning-rate 2e-5 \
  --random-seed 765 \
  --output-dir "${OUTPUT_DIR}"
