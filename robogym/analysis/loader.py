"""Load runner result CSVs into tidy records.

Reads the CSV tree
(``global_summary.csv`` / ``suite_summary.csv`` / ``task_*/task_details.csv``)
produced by :mod:`robogym.runners` so the analysis layer can tabulate/plot a
real eval run the same way it plots the paper's transcribed numbers.
"""

from __future__ import annotations

import csv
from pathlib import Path

def load_global_summary(run_dir: str | Path) -> dict:
    """Return the single suite-level row of ``global_summary.csv`` as a dict."""
    p = Path(run_dir) / "global_summary.csv"
    with open(p) as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}

def load_suite_summary(run_dir: str | Path) -> list[dict]:
    """Return per-task rows of ``suite_summary.csv``."""
    p = Path(run_dir) / "suite_summary.csv"
    with open(p) as f:
        return list(csv.DictReader(f))

def runs_to_model_table(run_dirs: dict[str, str | Path]) -> dict[str, list]:
    """``{model_name: run_dir}`` -> ``{model_name: [SR,Comp,Space,Time,
    Smooth,Safety]}`` in the paper's Table-2/3 column order."""
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
