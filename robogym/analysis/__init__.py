"""RoboGym analysis: tables, figures, and correlations.

Renders Tables 2-5 and the Figure 3 radar charts from either the reported
paper numbers in ``paper_results`` or a live runner's CSV tree, using the
same code paths in both cases. The bar/heatmap helpers are supporting
diagnostics.
"""

from .correlation import (
    kendalls_w,
    metric_correlation_matrix,
    spearman_vs_human,
)
from .human_study import HumanStudy
from .figures import (
    category_bars,
    heatmap,
    metric_bars,
    radar_chart,
    render_paper_figures,
    reproduce_paper_figures,
)
from .loader import (
    load_global_summary,
    load_suite_summary,
    runs_to_model_table,
)
from .tables import (
    model_table_to_csv,
    model_table_to_markdown,
    render_paper_tables,
    reproduce_paper_tables,
)

__all__ = [
    "metric_correlation_matrix", "kendalls_w", "spearman_vs_human",
    "HumanStudy",
    "radar_chart", "metric_bars", "category_bars", "heatmap",
    "render_paper_figures", "reproduce_paper_figures",
    "load_global_summary", "load_suite_summary", "runs_to_model_table",
    "model_table_to_markdown", "model_table_to_csv",
    "render_paper_tables", "reproduce_paper_tables",
]
