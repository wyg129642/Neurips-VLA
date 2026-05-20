#!/usr/bin/env bash
# Paper §4.1 fine-tuning step for π0 / π0.5 on the RoboGym expert dataset,
# using the surviving pi05/openpi training entrypoint + LeRobot ingestion.
# Full-stack only:  pip install -e ".[sim]"  + GPU + π0.5 base checkpoint.
#
#   scripts/finetune_pi05.sh <lerobot_dataset_repo_or_path> <exp_name>
set -euo pipefail
DATA="${1:?usage: finetune_pi05.sh <lerobot_dataset> <exp_name>}"
EXP="${2:-robogym_pi05}"
cd "$(dirname "$0")/../pi05/openpi"

# Convert the RoboGym RLDS shards to a LeRobot dataset (openpi ingests
# LeRobot), then launch openpi training (config name per openpi/src/openpi).
python scripts/compute_norm_stats.py --config-name pi05_libero || true
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py \
  --config-name pi05_libero \
  --exp-name "$EXP" \
  --data.repo-id "$DATA"
