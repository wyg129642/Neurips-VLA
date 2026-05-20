"""Figure 3 radar charts and supporting bar / heatmap diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_RADAR_AXES = ["SR", "Comp", "Space", "Time", "Smooth", "Safety"]


def _norm_row(row):
    """SR is 0-1, the rest 0-100 -> put SR on the same 0-100 radial scale."""
    r = list(row)
    return [r[0] * 100.0] + r[1:]


def radar_chart(model_table: dict[str, list], save_path: str | Path,
                title: str = "Multi-dimensional performance") -> Path:
    labels = _RADAR_AXES
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    ang = np.concatenate([ang, ang[:1]])

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for model, row in model_table.items():
        vals = _norm_row(row)
        vals = np.concatenate([vals, vals[:1]])
        ax.plot(ang, vals, lw=1.6, label=model)
        ax.fill(ang, vals, alpha=0.05)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_title(title, pad=24, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.10), fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return Path(save_path)


def metric_bars(model_table: dict[str, list], save_path: str | Path,
                title: str = "Performance comparison across six metrics") -> Path:
    models = list(model_table)
    fig, axs = plt.subplots(2, 3, figsize=(17, 9))
    for mi, mname in enumerate(_RADAR_AXES):
        ax = axs[mi // 3][mi % 3]
        vals = [model_table[m][mi] for m in models]
        order = np.argsort(vals)[::-1]
        om = [models[i] for i in order]
        ov = [vals[i] for i in order]
        ax.bar(range(len(om)), ov, color="#4477aa")
        for i, v in enumerate(ov):
            ax.text(i, v, f"{v:.2f}" if mi == 0 else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(om)))
        ax.set_xticklabels(om, rotation=40, ha="right", fontsize=7)
        ax.set_title(mname)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return Path(save_path)


def category_bars(by_cat: dict[str, list], cat_names: list[str],
                  save_path: str | Path) -> Path:
    models = list(by_cat)
    x = np.arange(len(models))
    w = 0.8 / len(cat_names)
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#4477aa", "#ee6677", "#228833", "#ccbb44"]
    for c, cname in enumerate(cat_names):
        vals = [by_cat[m][c] for m in models]
        ax.bar(x + c * w, vals, w, label=cname, color=colors[c % len(colors)])
        for i, v in enumerate(vals):
            ax.text(x[i] + c * w, v, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=6)
    ax.set_xticks(x + w * (len(cat_names) - 1) / 2)
    ax.set_xticklabels(models, rotation=35, ha="right")
    ax.set_ylabel("Success Rate")
    ax.set_title("Model success rates across reasoning categories")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return Path(save_path)


def heatmap(matrix, row_labels, col_labels, save_path: str | Path,
            title: str = "", cbar_label: str = "", fmt: str = "{:.2f}") -> Path:
    M = np.asarray(matrix, float)
    fig, ax = plt.subplots(figsize=(1.4 * len(col_labels) + 3,
                                    0.6 * len(row_labels) + 2))
    im = ax.imshow(M, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    color="white" if v > (M.max() + M.min()) / 2 else "black",
                    fontsize=8)
    cb = fig.colorbar(im)
    if cbar_label:
        cb.set_label(cbar_label)
    if title:
        ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return Path(save_path)


def render_paper_figures(out_dir: str | Path) -> list[Path]:
    """Render Figure 3 plus the Table 4-5 diagnostic plots."""
    import sys

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from paper_results import paper_tables as P

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return [
        radar_chart(P.TABLE2_AUGMENTED_LIBERO,
                    out / "fig3a_radar_augmented_libero.png",
                    "Figure 3a: Augmented LIBERO"),
        radar_chart(P.TABLE3_REASONING,
                    out / "fig3b_radar_reasoning.png",
                    "Figure 3b: System 2 reasoning tasks"),
        metric_bars(P.TABLE2_AUGMENTED_LIBERO,
                    out / "diag_bars_augmented_libero.png",
                    "Augmented LIBERO (Table 2)"),
        metric_bars(P.TABLE3_REASONING,
                    out / "diag_bars_reasoning.png",
                    "50 System 2 reasoning tasks (Table 3)"),
        category_bars(P.TABLE4_BY_CATEGORY, P.CATEGORY_NAMES,
                      out / "diag_table4_category_bars.png"),
        heatmap(P.TABLE5_SPEARMAN, P.TABLE5_METRIC_NAMES,
                P.TABLE5_METRIC_NAMES,
                out / "diag_table5_spearman_heatmap.png",
                "Table 5: Spearman rho of five key metrics",
                "Spearman rho"),
    ]


reproduce_paper_figures = render_paper_figures
