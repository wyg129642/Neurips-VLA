#!/usr/bin/env bash
# OpenVLA-OFT LIBERO eval with multi-dimensional metrics.
# Requires the full sim stack:  pip install -e ".[sim]"  + OpenVLA-OFT weights.
set -euo pipefail

CKPT="${1:?usage: run_openvla_oft.sh <pretrained_checkpoint> [task_suite] [datasets_root]}"
SUITE="${2:-libero_object}"
DATASETS_ROOT="${3:-./LIBERO/datasets}"

cd "$(dirname "$0")/../openvla-oft"

python experiments/robot/libero/run_libero_eval_custom_metrics.py \
  --pretrained_checkpoint "$CKPT" \
  --task_suite_name "$SUITE" \
  --datasets_root "$DATASETS_ROOT" \
  --use_expert_init_state True \
  --use_custom_metrics True \
  --num_trials_per_task 10
