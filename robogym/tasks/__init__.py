"""RoboGym System-2 reasoning task suite (paper §3.6 / Fig 2 / Table 4).

50 deterministically-seeded tasks across 3 levels, all driven through the same
unified runner and trajectory evaluator as augmented LIBERO, following the
representative-task descriptions in §3.6.
"""

from .base import CATEGORIES, ReasoningTask
from .memory_sequential.sim import (
    ColorHanoiSim,
    ColorHanoiTask,
    CountingSim,
    SequentialCountingTask,
)
from .physical_intuition.sim import (
    MazeSim,
    MazeTask,
    SeesawSim,
    SeesawWeightTask,
)
from .registry import build_reasoning_suite, reasoning_suite_by_category
from .symbolic_geometric.sim import (
    NumberBlockSim,
    NumberBlockTask,
    TangramSim,
    TangramTask,
)

__all__ = [
    "ReasoningTask", "CATEGORIES",
    "MazeTask", "SeesawWeightTask",
    "TangramTask", "NumberBlockTask",
    "ColorHanoiTask", "SequentialCountingTask",
    "MazeSim", "SeesawSim", "TangramSim", "NumberBlockSim",
    "ColorHanoiSim", "CountingSim",
    "build_reasoning_suite", "reasoning_suite_by_category",
]
