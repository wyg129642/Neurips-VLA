"""RoboGym multi-dimensional metric engine.

- :class:`TrajectoryEvaluator`: the streaming heuristic evaluator.
- :func:`score_episode`: paper-faithful §3.2-3.5 scorer (drop-in).
- :mod:`dtw`, :mod:`sparc`, :mod:`safety`: canonical §3.2 / §3.4 / §3.5
  implementations.
- :mod:`aggregate`: CSV schema for task, suite, and global summaries.
"""

from .aggregate import (
    append_suite_row,
    init_summary_csvs,
    summarize_task,
    write_global_row,
    write_task_details,
)
from .paper_metrics import score_episode
from .trajectory_evaluator import TrajectoryEvaluator

__all__ = [
    "TrajectoryEvaluator",
    "score_episode",
    "write_task_details",
    "summarize_task",
    "init_summary_csvs",
    "append_suite_row",
    "write_global_row",
]
