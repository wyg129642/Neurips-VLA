"""CLI: build the balanced expert dataset.

    python -m robogym.data.build_dataset --demos-per-task 500 --out datasets/robogym_expert
    python -m robogym.data.build_dataset --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .export import write_manifest, write_rlds_shard, write_task_dataset
from .trajectory_generator import GenConfig, generate_dataset


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="RoboGym expert dataset builder")
    ap.add_argument("--out", default="datasets/robogym_expert")
    ap.add_argument("--demos-per-task", type=int, default=500)
    ap.add_argument("--suite-seed", type=int, default=0)
    ap.add_argument("--jitter", type=float, default=0.012)
    ap.add_argument("--rlds", action="store_true",
                    help="also emit RLDS/TFDS shards")
    ap.add_argument("--quick", action="store_true",
                    help="smoke build: 3 demos per task")
    args = ap.parse_args(argv)

    cfg = GenConfig(
        demos_per_task=3 if args.quick else args.demos_per_task,
        suite_seed=args.suite_seed, jitter=args.jitter)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Building expert dataset -> {out}  "
          f"(target {cfg.demos_per_task} demos/task x 50 tasks "
          f"= {cfg.demos_per_task * 50} demos)")

    final_report = None
    for task, demos, report in generate_dataset(cfg):
        info = write_task_dataset(out / task.category, task.name, demos)
        if args.rlds:
            write_rlds_shard(out, task.name, demos)
        print(f"  [{task.category:9s}] {task.name:26s} "
              f"{info['num_demos']:4d} demos -> {info['format']}")
        final_report = report

    man = write_manifest(out, final_report, cfg)
    print(f"\nTotal demos: {final_report.total_demos} | "
          f"per-category: {final_report.per_category} | "
          f"balanced: {len(final_report.shortfall) == 0}")
    print(f"Manifest: {man}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
