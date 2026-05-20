"""Read runner CSV trees into model tables."""

from __future__ import annotations

import csv
from pathlib import Path


def load_global_summary(run_dir: str | Path) -> dict:
    p = Path(run_dir) / "global_summary.csv"
    with open(p) as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def load_suite_summary(run_dir: str | Path) -> list[dict]:
    p = Path(run_dir) / "suite_summary.csv"
    with open(p) as f:
        return list(csv.DictReader(f))


def runs_to_model_table(run_dirs: dict[str, str | Path]) -> dict[str, list]:
    """``{model: run_dir}`` -> ``{model: [SR, Comp, Space, Time, Smooth, Safety]}``."""
    table = {}
    for model, rd in run_dirs.items():
        g = load_global_summary(rd)
        if not g:
            continue
        table[model] = [
            float(g["success_rate"]),
            float(g["avg_completion"]),
            float(g["avg_space_eff"]),
            float(g["avg_time_eff"]),
            float(g["avg_smoothness"]),
            float(g["avg_safety"]),
        ]
    return table
