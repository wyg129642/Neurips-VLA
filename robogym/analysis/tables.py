"""Render Tables 2-6 from the transcribed paper numbers or a live runner."""

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
    """Render Tables 2-6 as CSV and a combined markdown report."""
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
        "# RoboGym: paper Tables 2-6 (rendered from reported numbers)\n",
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
    md += [
        "",
        "**Table 5: Spearman rho of five key metrics "
        "(System 2 reasoning tasks)**\n",
        "| | " + " | ".join(P.TABLE5_METRIC_NAMES) + " |",
        "|---|" + "---|" * len(P.TABLE5_METRIC_NAMES),
    ]
    for name, row in zip(P.TABLE5_METRIC_NAMES, P.TABLE5_SPEARMAN):
        md.append(f"| {name} | "
                  + " | ".join(f"{v:.2f}" for v in row) + " |")
    md += [
        "",
        "**Table 6 (Appendix B): Spearman rho between automated metrics and "
        "three-expert human consensus**",
        "",
        "| Dimension | " + " | ".join(P.TABLE6_HUMAN_AUTOMATION.keys())
        + " | mean |",
        "|---|" + "---|" * (len(P.TABLE6_HUMAN_AUTOMATION) + 1),
        "| Spearman rho | "
        + " | ".join(f"{v:.2f}" for v in P.TABLE6_HUMAN_AUTOMATION.values())
        + f" | {P.MEAN_SPEARMAN:.2f} |",
        "",
        f"_Inter-rater agreement across {P.HUMAN_STUDY_N_EXPERTS} experts and "
        f"{P.HUMAN_STUDY_N_MODELS} models: Kendall's W = "
        f"{P.INTER_RATER_KENDALLS_W} (p < 0.001)._",
    ]
    report = out / "paper_tables_rendered.md"
    report.write_text("\n".join(md))
    made["report"] = report
    return made


reproduce_paper_tables = render_paper_tables
