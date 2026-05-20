"""50-task System-2 reasoning suite (Sec. 3.6, Figure 2)."""

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
