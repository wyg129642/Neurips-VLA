"""LIBERO simulation backend with the Figure-1 task-augmentation wrapper.

Adapts a robosuite/mujoco LIBERO env to :class:`robogym.envs.base.SimBackend`
so the streaming metric engine runs unchanged on real LIBERO. The
augmentation wrapper applies the Figure 1 / §3.1 pipeline.

Imports of ``libero`` and ``robosuite`` are deferred so the rest of the
package (metrics, analysis, synthetic demo, tests) is usable on machines
without the heavy simulation stack.
"""

from __future__ import annotations

import numpy as np

from .base import RoboGymEnv

class LiberoBackend:
    """Wraps a robosuite ``env`` (LIBERO) in the :class:`SimBackend` protocol.

    The few simulator operations the trajectory evaluator needs map 1:1 onto the
    mujoco ``env.sim`` API the original implementation used directly.
    """

    def __init__(self, env, ee_site_name: str = "gripper0_grip_site"):
        self.env = env
        self.sim = env.sim
        self._ee_site_id = self.sim.model.site_name2id(ee_site_name)

    @property
    def n_bodies(self) -> int:
        return self.sim.model.nbody

    def body_names(self) -> list[str]:
        return [self.sim.model.body_id2name(i) for i in range(self.n_bodies)]

    def ee_pos(self) -> np.ndarray:
        return self.sim.data.site_xpos[self._ee_site_id].copy()

    def ee_velocity(self) -> np.ndarray:
        return self.sim.data.site_xvelp[self._ee_site_id].copy()

    def body_xpos(self, body_id: int) -> np.ndarray:
        return self.sim.data.body_xpos[body_id].copy()

    def get_state(self) -> np.ndarray:
        return self.sim.get_state().flatten()

    def set_state(self, flat_state: np.ndarray) -> None:
        self.sim.set_state_from_flattened(flat_state)

    def forward(self) -> None:
        self.sim.forward()

    def reset(self) -> dict:
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def contact_forces(self) -> np.ndarray:
        return self.sim.data.cfrc_ext[1:]

def make_libero_env(task, model_family: str = "openvla",
                    resolution: int = 256) -> tuple[object, str]:
    """Construct a LIBERO env via the ``libero_utils.get_libero_env``.

    Returns ``(env, task_description)``. Raises a clear error if the simulation
    stack is unavailable (machines without the heavy stack) so callers can fall back to the
    synthetic backend.
    """
    try:
        from experiments.robot.libero.libero_utils import get_libero_env
    except Exception as exc:  # pragma: no cover - depends on full stack
        raise RuntimeError(
            "Real LIBERO unavailable (need robosuite/mujoco/libero + the "
            "openvla-oft experiments package on PYTHONPATH). Use the synthetic "
            "backend for a dependency-free run: see robogym.envs.synthetic_env."
        ) from exc
    return get_libero_env(task, model_family, resolution=resolution)

def build_augmented_env(task, augmentation: dict | None = None,
                        model_family: str = "openvla",
                        resolution: int = 256, dt: float = 0.05) -> RoboGymEnv:
    """The paper's "Task Augmentation" step (Fig 1): take a standard
    binary-success LIBERO task and return a :class:`RoboGymEnv` carrying the
    multi-dimensional scoring metadata produced by the augmentation pipeline.
    """
    env, desc = make_libero_env(task, model_family, resolution)
    return RoboGymEnv(LiberoBackend(env), task_description=desc,
                      augmentation=augmentation, dt=dt)
