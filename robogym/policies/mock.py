"""Scripted surrogate policy with tunable competence and failure modes.

Used by tests and the demo to exercise the metric engine without loading a
real VLA. The output is the expert action plus competence-scaled meander,
jitter, sloth, occasional force violations, and an optional early-abandon
event, so the trajectory evaluator produces a believable spread across the
five dimensions. ``competence`` in ``[0, 1]`` interpolates between random
behaviour and near-expert behaviour.
"""

from __future__ import annotations

import numpy as np

from .base import Policy

class MockPolicy(Policy):
    def __init__(self, competence: float = 0.6, seed: int = 0,
                 meander: float = 1.0, jitter: float = 1.0,
                 sloth: float = 1.0, force_violations: float = 1.0,
                 drop_prob: float = 0.0, name: str | None = None):
        self.c = float(np.clip(competence, 0.0, 1.0))
        self.rng = np.random.default_rng(seed)
        self.meander = meander
        self.jitter = jitter
        self.sloth = sloth
        self.force_violations = force_violations
        self.drop_prob = drop_prob
        self.name = name or f"mock_c{self.c:.2f}"
        self._actions: np.ndarray | None = None
        self._t = 0
        self._drop_at: int | None = None

    def reset(self, *, expert_actions=None, expert_init_state=None,
              task_description="", **kw) -> None:
        self._actions = (None if expert_actions is None
                         else np.asarray(expert_actions, dtype=float))
        self._t = 0
        self._drop_at = None
        if (self._actions is not None and self.drop_prob > 0
                and self.rng.random() < self.drop_prob):
            # abandon the task partway -> exercises completeness/"dropped"
            self._drop_at = int(len(self._actions)
                                * self.rng.uniform(0.35, 0.7))

    def act(self, observation: dict, task_description: str) -> np.ndarray:
        if self._actions is None or self._t >= len(self._actions):
            return np.zeros(7)
        if self._drop_at is not None and self._t >= self._drop_at:
            self._t += 1
            return np.zeros(7)  # froze / gave up

        a = self._actions[self._t].copy()
        self._t += 1
        incompetence = 1.0 - self.c

        # meander: low-freq lateral drift -> hurts spatial efficiency
        drift = self.meander * incompetence * 0.45 * np.sin(self._t * 0.22)
        a[:3] += drift

        # jitter: high-freq tremor -> hurts smoothness (FFT/SPARC)
        a[:6] += (self.jitter * incompetence * 0.6
                  * self.rng.normal(0, 1, size=min(6, len(a))))

        # sloth: shrink the commanded step -> hurts temporal efficiency
        a[:3] *= (1.0 - 0.55 * self.sloth * incompetence)

        # force violations: occasionally drive hard into the table -> safety
        if (self.force_violations > 0
                and self.rng.random() < 0.05 * incompetence):
            a[2] -= 1.0 * self.force_violations
        return a
