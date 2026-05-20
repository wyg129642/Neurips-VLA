"""Tables, figures, and correlation analyses."""

from .correlation import (
    kendalls_w,
    metric_correlation_matrix,
    spearman_vs_human,
)
from .figures import (
    category_bars,
    heatmap,
    metric_bars,
    radar_chart,
    render_paper_figures,
    reproduce_paper_figures,
)
from .human_study import HumanStudy
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
