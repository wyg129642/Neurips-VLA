#!/usr/bin/env bash
# π0 / π0.5 multi-dimensional eval via the unified runner + trajectory evaluator.
#
# 1. In one shell, serve the model from pi05/openpi:
#       cd pi05/openpi && python scripts/serve_policy.py --env LIBERO
# 2. Then run this (full sim stack required: pip install -e ".[sim]").
set -euo pipefail

SUITE="${1:-libero_object}"
HOST="${2:-0.0.0.0}"
PORT="${3:-8000}"
DATASETS_ROOT="${4:-./LIBERO/datasets}"

cd "$(dirname "$0")/.."

python - "$SUITE" "$HOST" "$PORT" "$DATASETS_ROOT" <<'PY'
import sys
from robogym.policies import OpenPiPolicy
from robogym.runners import EvalConfig, run_suite

suite, host, port, root = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
cfg = EvalConfig(backend="libero", scorer="streaming",
                 task_suite_name=suite, datasets_root=root,
                 num_trials_per_task=10, results_dir="./results/pi05")
run_suite(cfg, OpenPiPolicy(host=host, port=port, model_name="pi05"))
PY
