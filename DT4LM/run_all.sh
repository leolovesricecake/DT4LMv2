set -euo pipefail

DATASET="$1"
CUDA_DEVICES="$2"
# MODEL_ID="${3:albertbasev1-v2}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" bash experiments/improvements/run_first_round.sh \
  experiments/improvements/configs/${DATASET}/albertbasev1-v2-base.yaml

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" bash experiments/improvements/run_first_round.sh \
  experiments/improvements/configs/${DATASET}/albertbasev1-v2-ff-pbs.yaml

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" bash experiments/improvements/run_first_round.sh \
  experiments/improvements/configs/${DATASET}/albertbasev1-v2-dynamic-beam.yaml

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" bash experiments/improvements/run_first_round.sh \
  experiments/improvements/configs/${DATASET}/albertbasev1-v2-ff-pareto-greedy.yaml

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" bash experiments/improvements/run_first_round.sh \
  experiments/improvements/configs/${DATASET}/albertbasev1-v2-hard-pbs.yaml

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" bash experiments/improvements/run_first_round.sh \
  experiments/improvements/configs/${DATASET}/albertbasev1-v2-ff-mnew.yaml

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" bash experiments/improvements/run_first_round.sh \
  experiments/improvements/configs/${DATASET}/albertbasev1-v2-ff-pbs-k3.yaml

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" bash experiments/improvements/run_first_round.sh \
  experiments/improvements/configs/${DATASET}/albertbasev1-v2-ff-pbs-k10.yaml
