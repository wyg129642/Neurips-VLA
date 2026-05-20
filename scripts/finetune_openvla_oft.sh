#!/usr/bin/env bash
# Paper §4.1 fine-tuning step for OpenVLA-OFT on the RoboGym expert dataset.
# Uses the openvla-oft LoRA finetune entrypoint.
# Full-stack only:  pip install -e ".[sim]"  + GPU + OpenVLA-OFT base weights.
#
#   scripts/finetune_openvla_oft.sh <rlds_data_root> <run_name>
set -euo pipefail
DATA_ROOT="${1:?usage: finetune_openvla_oft.sh <rlds_data_root> <run_name>}"
RUN_NAME="${2:-robogym_oft}"
cd "$(dirname "$0")/../openvla-oft"

# RoboGym demos are emitted in the LIBERO HDF5 schema; convert to
# RLDS exactly as the repo expects (see openvla-oft LIBERO.md), or
# point --data_root_dir at the RLDS shards from generate_expert_dataset.sh.
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir "$DATA_ROOT" \
  --dataset_name robogym_reasoning \
  --run_root_dir ./runs \
  --use_l1_regression True \
  --use_proprio True \
  --num_images_in_input 2 \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --run_id_note "$RUN_NAME"
