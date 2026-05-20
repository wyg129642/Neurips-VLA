"""Backend abstraction for the RoboGym metric engine.

The ``TrajectoryEvaluator`` (see :mod:`robogym.metrics.trajectory_evaluator`)
was written directly against the robosuite/mujoco ``env.sim`` API. To make the *exact
same algorithm* runnable both on real LIBERO and on the dependency-free synthetic
backend, the few simulator operations it needs are factored into this protocol.

A backend exposes just enough of a mujoco-like sim:

* end-effector cartesian position,
* rigid-body names + positions (for tracking moved objects),
* flattened state save/restore + ``forward`` (for the expert pre-roll),
* ``step(action)`` returning ``(obs, reward, done, info)``,
* per-body external contact force magnitudes (``cfrc_ext`` analogue).

``RoboGymEnv`` is the user-facing env: a backend + task spec + the augmentation
metadata produced by the pipeline (Fig 1 of the paper).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

@runtime_checkable
class SimBackend(Protocol):
    """Minimal mujoco-like simulator surface used by the metric engine."""

    # topology
    @property
    def n_bodies(self) -> int:
        """Number of rigid bodies."""

    def body_names(self) -> list[str]:
        """Ordered list of body names (index == body id)."""

    # kinematics
    def ee_pos(self) -> np.ndarray:
        """Current end-effector (gripper) cartesian position, shape (3,)."""

    def body_xpos(self, body_id: int) -> np.ndarray:
        """World position of body ``body_id``, shape (3,)."""

    def ee_velocity(self) -> np.ndarray:
        """Current end-effector cartesian velocity, shape (3,). Optional -
        the evaluator finite-differences positions if this is unavailable."""

    # state management
    def get_state(self) -> np.ndarray:
        """Flattened simulator state (for save/restore around expert pre-roll)."""

    def set_state(self, flat_state: np.ndarray) -> None:
        """Restore a flattened simulator state."""

    def forward(self) -> None:
        """Recompute derived quantities after a state set (mujoco ``forward``)."""

    # dynamics
    def step(self, action: Any) -> tuple[dict, float, bool, dict]:
        """Advance one control step. Returns ``(obs, reward, done, info)``."""

    def reset(self) -> dict:
        """Reset and return the first observation."""

    def contact_forces(self) -> np.ndarray:
        """Per-body external force magnitudes (mujoco ``cfrc_ext[1:]`` analogue),
        shape (n_bodies-1,). Used by the operational-safety metric."""

class RoboGymEnv:
    """A backend, task description, and augmentation metadata.

    This is the object a runner drives. It owns a :class:`SimBackend` and
    the augmentation record (phase weights, safe workspace, force limits)
    that the task-augmentation pipeline attaches per task. The record is
    the artifact produced by the Figure 1 pipeline.
    """

    def __init__(self, backend: SimBackend, task_description: str = "",
                 augmentation: dict | None = None, dt: float = 0.05):
        self.backend = backend
        self.task_description = task_description
        self.augmentation = augmentation or {}
        self.dt = dt

    # Convenience pass-throughs
    def reset(self) -> dict:
        return self.backend.reset()

    def step(self, action: Any) -> tuple[dict, float, bool, dict]:
        return self.backend.step(action)

    @property
    def sim(self) -> SimBackend:  # name parity with code paths
        return self.backend
