"""Registry for the System-2 reasoning suite (paper §3.6, Table 4).

Table 4 reports success per category (Geometric, Physical, Memory). This
registry instantiates 50 deterministically-seeded task instances spread
across six task families, so the reasoning suite runs through the same
unified runner and trajectory evaluator as augmented LIBERO.
"""

from __future__ import annotations

from .memory_sequential.tasks import ColorHanoiTask, SequentialCountingTask
from .physical_intuition.maze import MazeTask
from .physical_intuition.seesaw import SeesawWeightTask
from .symbolic_geometric.tasks import NumberBlockTask, TangramTask

# (family class, count) -- 18 + 18 + 14 = 50 instances.
_GEOMETRIC = [(TangramTask, 9), (NumberBlockTask, 9)]
_PHYSICAL = [(MazeTask, 9), (SeesawWeightTask, 9)]
_MEMORY = [(ColorHanoiTask, 7), (SequentialCountingTask, 7)]

_LEVELS = {"geometric": _GEOMETRIC, "physical": _PHYSICAL, "memory": _MEMORY}

def build_reasoning_suite(seed: int = 0) -> list:
    """Return 50 reasoning task instances (geometric, then physical, then
    memory), each with a unique deterministic randomization seed."""
    tasks, tid = [], 0
    for level in ("geometric", "physical", "memory"):
        for cls, count in _LEVELS[level]:
            for k in range(count):
                tasks.append(cls(task_id=tid, seed=seed * 9973 + tid))
                tid += 1
    assert len(tasks) == 50, f"expected 50 reasoning tasks, got {len(tasks)}"
    return tasks

def reasoning_suite_by_category(seed: int = 0) -> dict[str, list]:
    out: dict[str, list] = {"geometric": [], "physical": [], "memory": []}
    for t in build_reasoning_suite(seed):
        out[t.category].append(t)
    return out
