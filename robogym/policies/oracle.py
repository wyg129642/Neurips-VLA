"""Expert-replay oracle policy used as the metric-engine sanity baseline."""

from __future__ import annotations

import numpy as np

from .base import Policy


class OraclePolicy(Policy):
    name = "oracle"

    def __init__(self, noise: float = 0.0, seed: int = 0):
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self._actions: np.ndarray | None = None
        self._t = 0

    def reset(self, *, expert_actions=None, expert_init_state=None,
              task_description="", **kw) -> None:
        self._actions = (None if expert_actions is None
                         else np.asarray(expert_actions, dtype=float))
        self._t = 0

    def act(self, observation: dict, task_description: str) -> np.ndarray:
        if self._actions is None or self._t >= len(self._actions):
            return np.zeros(7)
        a = self._actions[self._t].copy()
        self._t += 1
        if self.noise > 0:
            a[:6] += self.rng.normal(0, self.noise, size=min(6, len(a)))
        return a
