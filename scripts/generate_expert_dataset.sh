#!/usr/bin/env bash
# Paper §4.1: generate 500 success-filtered expert demos/task → 25k balanced,
# exported in the LIBERO HDF5 schema (+ RLDS shards + manifest).
#
#   scripts/generate_expert_dataset.sh [demos_per_task] [out_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
DEMOS="${1:-500}"
OUT="${2:-datasets/robogym_expert}"
python -m robogym.data.build_dataset --demos-per-task "$DEMOS" --out "$OUT" --rlds
echo "Dataset ready at $OUT (consumable by the ExpertDataLoader and"
echo "the unified runner's backend=libero path; RLDS shards for VLA finetuning)."
