"""LIBERO backend (real robosuite/mujoco sim) and augmentation wrapper."""

from __future__ import annotations

import numpy as np

from .base import RoboGymEnv


class LiberoBackend:
    """Adapts a robosuite LIBERO env to the :class:`SimBackend` protocol."""

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
    try:
        from experiments.robot.libero.libero_utils import get_libero_env
    except Exception as exc:
        raise RuntimeError(
            "Real LIBERO unavailable (robosuite + mujoco + libero + the "
            "openvla-oft experiments package must be importable). Use the "
            "synthetic backend for a dependency-free run."
        ) from exc
    return get_libero_env(task, model_family, resolution=resolution)


def build_augmented_env(task, augmentation: dict | None = None,
                        model_family: str = "openvla",
                        resolution: int = 256, dt: float = 0.05) -> RoboGymEnv:
    """Wrap a LIBERO task in a :class:`RoboGymEnv` carrying the augmentation record."""
    env, desc = make_libero_env(task, model_family, resolution)
    return RoboGymEnv(LiberoBackend(env), task_description=desc,
                      augmentation=augmentation, dt=dt)
