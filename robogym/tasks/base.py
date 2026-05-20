"""System-2 reasoning task framework (paper §3.6 / Figure 2).

Each task is backed by a :class:`robogym.envs.physics_sim.PhysicsReasoningSim`
subclass: success follows from a physical/logical state (torque balance,
stacking legality, wall collision, containment count, geometric fit), the
reasoning cue is exposed in the observation, and the algorithmic oracle
(paper §4.1) is the only thing that knows the answer. The policy does not.
A non-reasoning baseline therefore fails on these tasks; see
``generate_naive_demo`` per sim for the controlled-baseline check.

Test-time randomization of task-critical variables (position, colour, mass,
target sequence) is implemented inside each sim's ``_setup`` and re-seeded
per episode, so memorising a single trajectory does not work.
"""

from __future__ import annotations

import numpy as np

from ..envs.base import RoboGymEnv

CATEGORIES = ("geometric", "physical", "memory")

class ReasoningTask:
    """Thin wrapper around a physics-grounded reasoning sim.

    Subclasses set :attr:`category` and :attr:`sim_cls` (a
    :class:`PhysicsReasoningSim`). Test-time randomization happens inside the
    sim's ``_setup`` (re-seeded per episode), so :meth:`randomize` just rebinds
    the seed and rebuilds the instance.
    """

    category: str = "geometric"
    sim_cls = None  # set by subclass
    family: str = "reasoning"

    def __init__(self, task_id: int = 0, seed: int = 0, dt: float = 0.05):
        self.task_id = task_id
        self.seed = seed
        self.dt = dt
        self.name = f"{self.family}_{task_id}"
        self._sim = self.sim_cls(seed=seed, dt=dt)
        self._instance = self._sim.instance

    def randomize(self, seed: int) -> dict:
        """Resample task-critical variables for a fresh episode (§3.6)."""
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
        """Algorithmic-oracle trajectory as ``(actions, init_state)``.

        Re-randomises when ``seed`` is given (test-time randomization), so two
        demos of the same task face different instances and a policy cannot
        memorise a fixed solution.
        """
        if seed is not None:
            self.randomize(seed)
        return self._sim.generate_expert_demo(seed=seed, jitter=jitter)

    # introspection used by tests / non-reasoning baseline
    @property
    def instance(self) -> dict:
        return self._instance
