"""RoboGym automated expert-trajectory pipeline (paper §4.1).

algorithmic-oracle -> B-spline-smoothed waypoints ->
success-filtered rollout -> 500 demos/task -> 25k balanced across logical
domains, exported in the LIBERO HDF5 schema (+ RLDS + manifest).
"""

from .export import (
    write_manifest,
    write_rlds_shard,
    write_task_dataset,
)
from .trajectory_generator import (
    Demo,
    GenConfig,
    GenReport,
    generate_dataset,
    generate_task_demos,
)

__all__ = [
    "GenConfig", "GenReport", "Demo",
    "generate_dataset", "generate_task_demos",
    "write_task_dataset", "write_rlds_shard", "write_manifest",
]
