"""Render the paper's Tables 2-5.

Emits markdown / CSV in the layout of the paper tables from either the
reported numbers in ``paper_results`` or from a runner's CSV tree via
:func:`robogym.analysis.loader.runs_to_model_table`. The §4.4
human-automation alignment study is rendered as a one-line summary
(reported mean Spearman ρ).
"""

from __future__ import annotations

import csv
from pathlib import Path

_COLS = ["Model", "SR", "Comp", "Space", "Time", "Smooth", "Safety"]

def model_table_to_markdown(model_table: dict[str, list],
                            caption: str = "") -> str:
    lines = []
    if caption:
        lines.append(f"**{caption}**\n")
    lines.append("| " + " | ".join(_COLS) + " |")
    lines.append("|" + "---|" * len(_COLS))
    for m, row in model_table.items():
        cells = [m, f"{row[0]:.2f}"] + [f"{v:.2f}" for v in row[1:]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

def model_table_to_csv(model_table: dict[str, list],
                       path: str | Path) -> Path:
    path = Path(path)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_COLS)
        for m, row in model_table.items():
            w.writerow([m, f"{row[0]:.2f}"] + [f"{v:.2f}" for v in row[1:]])
    return path

def render_paper_tables(out_dir: str | Path) -> dict[str, Path]:
    """Render Tables 2-5 to CSV and a combined markdown report from the
    transcribed paper numbers in :mod:`paper_results`.

    The numbers are taken from the published tables; this function does not
    re-run any models. To render from a live evaluation, build a
    ``model_table`` with :func:`robogym.analysis.loader.runs_to_model_table`
    and pass it to :func:`model_table_to_markdown` /
    :func:`model_table_to_csv` directly.
    """
    import sys

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from paper_results import paper_tables as P

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    made = {
        "table2": model_table_to_csv(P.TABLE2_AUGMENTED_LIBERO,
                                     out / "table2_augmented_libero.csv"),
        "table3": model_table_to_csv(P.TABLE3_REASONING,
                                     out / "table3_reasoning.csv"),
    }
    md = [
        "# RoboGym: paper Tables 2-5 (rendered from reported numbers)\n",
        model_table_to_markdown(P.TABLE2_AUGMENTED_LIBERO,
                                "Table 2: Performance on Augmented LIBERO"),
        "",
        model_table_to_markdown(P.TABLE3_REASONING,
                                "Table 3: Performance on System 2 reasoning tasks"),
        "",
        "**Table 4: success per reasoning category "
        "[Geometric, Physical, Memory]**\n",
        "| Model | Geometric | Physical | Memory |",
        "|---|---|---|---|",
    ]
    for m, r in P.TABLE4_BY_CATEGORY.items():
        md.append(f"| {m} | {r[0]:.2f} | {r[1]:.2f} | {r[2]:.2f} |")
    md += ["", "**Table 5: Spearman's ρ of five key metrics "
           "(System 2 reasoning tasks)**\n",
           "| | " + " | ".join(P.TABLE5_METRIC_NAMES) + " |",
           "|---|" + "---|" * len(P.TABLE5_METRIC_NAMES)]
    for name, row in zip(P.TABLE5_METRIC_NAMES, P.TABLE5_SPEARMAN):
        md.append(f"| {name} | "
                  + " | ".join(f"{v:.2f}" for v in row) + " |")
    md += ["",
           "_§4.4 Human-Automation Alignment: mean Spearman "
           f"ρ ~ {P.MEAN_SPEARMAN:.2f} between the "
           f"{P.HUMAN_STUDY_N_EXPERTS}-expert human consensus and the "
           f"automated metrics ({P.HUMAN_STUDY_N_MODELS} models, "
           f"{P.HUMAN_STUDY_N_DIMENSIONS} dimensions); per-dimension ρ and "
           "the inter-rater Kendall's W are produced by the study harness._"]
    report = out / "paper_tables_rendered.md"
    report.write_text("\n".join(md))
    made["report"] = report
    return made

# Backwards-compatible alias for callers that imported the previous name.
reproduce_paper_tables = render_paper_tables
