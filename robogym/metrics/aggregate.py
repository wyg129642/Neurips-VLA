"""Result aggregation into task / suite / global CSVs.

The CSV column schema and rounding match the OpenVLA-OFT runner
(:mod:`robogym.runners.run_libero_eval_custom_metrics`), so output is
interchangeable with that runner's ``task_details``, ``suite_summary``,
and ``global_summary`` files.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

METRIC_COLS = [
    "total_fwdbias", "total_hybrid", "completion", "space_eff", "time_eff",
    "smoothness", "safety", "raw_dev", "raw_space_val", "raw_time_val",
    "raw_fft_val", "dropped",
]
TASK_DETAIL_HEADER = ["task_id", "task_name", "task_description", "demo_idx",
                      "success"] + METRIC_COLS
SUITE_HEADER = ["suite_name", "task_id", "task_name", "num_episodes",
                "success_rate", "avg_total_fwdbias", "avg_completion",
                "avg_space_eff", "avg_time_eff", "avg_smoothness",
                "avg_safety", "avg_raw_dev", "avg_dropped"]
GLOBAL_HEADER = ["suite_name", "num_tasks", "num_episodes", "success_rate",
                 "avg_total_fwdbias", "avg_completion", "avg_space_eff",
                 "avg_time_eff", "avg_smoothness", "avg_safety",
                 "avg_raw_dev", "avg_dropped_rate"]

def _mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [r.get(key, 0.0) for r in rows if r.get(key, None) is not None]
    return float(np.mean(vals)) if vals else 0.0

def write_task_details(task_dir: Path, episode_rows: list[dict]) -> Path:
    """Write one task's per-episode ``task_details.csv`` (schema)."""
    task_dir.mkdir(parents=True, exist_ok=True)
    out = task_dir / "task_details.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TASK_DETAIL_HEADER)
        w.writeheader()
        for r in episode_rows:
            row = {k: r.get(k, 0.0 if k != "dropped" else False)
                   for k in TASK_DETAIL_HEADER}
            w.writerow(row)
    return out

def summarize_task(suite_name: str, task_id: int, task_name: str,
                   task_description: str, episode_rows: list[dict]) -> dict:
    """Per-task summary record (feeds suite_summary.csv)."""
    s = {
        "suite_name": suite_name,
        "task_id": task_id,
        "task_name": task_name,
        "task_description": task_description,
        "num_episodes": len(episode_rows),
        "success_rate": float(np.mean([r["success"] for r in episode_rows]))
        if episode_rows else 0.0,
    }
    for k in METRIC_COLS:
        s[f"avg_{k}"] = _mean(episode_rows, k)
    return s

def init_summary_csvs(out_dir: Path) -> tuple[Path, Path]:
    """Create ``global_summary.csv`` + ``suite_summary.csv`` with headers."""
    out_dir.mkdir(parents=True, exist_ok=True)
    g, s = out_dir / "global_summary.csv", out_dir / "suite_summary.csv"
    with open(g, "w", newline="") as f:
        csv.writer(f).writerow(GLOBAL_HEADER)
    with open(s, "w", newline="") as f:
        csv.writer(f).writerow(SUITE_HEADER)
    return g, s

def append_suite_row(suite_csv: Path, task_summary: dict) -> None:
    with open(suite_csv, "a", newline="") as f:
        csv.writer(f).writerow([
            task_summary["suite_name"], task_summary["task_id"],
            task_summary["task_name"], task_summary["num_episodes"],
            round(task_summary["success_rate"], 4),
            round(task_summary["avg_total_fwdbias"], 2),
            round(task_summary["avg_completion"], 2),
            round(task_summary["avg_space_eff"], 2),
            round(task_summary["avg_time_eff"], 2),
            round(task_summary["avg_smoothness"], 2),
            round(task_summary["avg_safety"], 2),
            round(task_summary["avg_raw_dev"], 3),
            round(task_summary["avg_dropped"], 4),
        ])

def write_global_row(global_csv: Path, suite_name: str,
                     per_task: list[dict]) -> dict:
    """Aggregate per-task summaries -> one suite-level ``global_summary.csv`` row."""
    if not per_task:
        return {}
    row = {
        "suite_name": suite_name,
        "num_tasks": len(per_task),
        "num_episodes": int(np.sum([t["num_episodes"] for t in per_task])),
        "success_rate": float(np.mean([t["success_rate"] for t in per_task])),
        "avg_total_fwdbias": float(np.mean([t["avg_total_fwdbias"]
                                            for t in per_task])),
        "avg_completion": float(np.mean([t["avg_completion"]
                                         for t in per_task])),
        "avg_space_eff": float(np.mean([t["avg_space_eff"]
                                        for t in per_task])),
        "avg_time_eff": float(np.mean([t["avg_time_eff"] for t in per_task])),
        "avg_smoothness": float(np.mean([t["avg_smoothness"]
                                         for t in per_task])),
        "avg_safety": float(np.mean([t["avg_safety"] for t in per_task])),
        "avg_raw_dev": float(np.mean([t["avg_raw_dev"] for t in per_task])),
        "avg_dropped_rate": float(np.mean([t["avg_dropped"]
                                           for t in per_task])),
    }
    with open(global_csv, "a", newline="") as f:
        csv.writer(f).writerow([
            row["suite_name"], row["num_tasks"], row["num_episodes"],
            round(row["success_rate"], 4),
            round(row["avg_total_fwdbias"], 2),
            round(row["avg_completion"], 2), round(row["avg_space_eff"], 2),
            round(row["avg_time_eff"], 2), round(row["avg_smoothness"], 2),
            round(row["avg_safety"], 2), round(row["avg_raw_dev"], 3),
            round(row["avg_dropped_rate"], 4),
        ])
    return row
