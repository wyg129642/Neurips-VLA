"""Reasoning-task wrapper for physics-grounded sims."""

from __future__ import annotations

import numpy as np

from ..envs.base import RoboGymEnv

CATEGORIES = ("geometric", "physical", "memory")


class ReasoningTask:
    """Thin wrapper around a :class:`PhysicsReasoningSim` subclass."""

    category: str = "geometric"
    sim_cls = None
    family: str = "reasoning"

    def __init__(self, task_id: int = 0, seed: int = 0, dt: float = 0.05):
        self.task_id = task_id
        self.seed = seed
        self.dt = dt
        self.name = f"{self.family}_{task_id}"
        self._sim = self.sim_cls(seed=seed, dt=dt)
        self._instance = self._sim.instance

    def randomize(self, seed: int) -> dict:
        """Resample task-critical variables for a fresh episode."""
        self.seed = seed
        self._sim = self.sim_cls(seed=seed, dt=self.dt)
        self._instance = self._sim.instance
        return self._instance

    @property
    def description(self) -> str:
        return self._instance.get("description", self.name)

    def make_sim(self):
        return self._sim

    def make_env(self) -> RoboGymEnv:
        return RoboGymEnv(self._sim, task_description=self.description,
                          dt=self.dt)

    def expert_demo(self, seed: int | None = None,
                    jitter: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        if seed is not None:
            self.randomize(seed)
        return self._sim.generate_expert_demo(seed=seed, jitter=jitter)

    @property
    def instance(self) -> dict:
        return self._instance
