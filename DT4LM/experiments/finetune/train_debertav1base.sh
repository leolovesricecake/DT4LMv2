#!/usr/bin/env bash
set -euo pipefail

DATASET="$1"
CUDA_DEVICES="$2"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DATASET_PATH="${PROJECT_ROOT}/outputs/datasets/${DATASET}"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/finetuned/debertav1base_${DATASET}"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" textattack train \
  --model-name-or-path /mnt/huawei/nsq/models/microsoft/deberta-base \
  --dataset "${DATASET_PATH}" \
  --model-max-length 256 \
  --per-device-train-batch-size 32 \
  --per-device-eval-batch-size 128 \
  --epochs 5 \
  --learning-rate 1e-5 \
  --random-seed 765 \
  --output-dir "${OUTPUT_DIR}"
