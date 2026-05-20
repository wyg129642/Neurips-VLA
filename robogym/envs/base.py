"""Backend protocol and the user-facing :class:`RoboGymEnv` wrapper."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SimBackend(Protocol):
    """Minimal mujoco-like simulator surface used by the metric engine."""

    @property
    def n_bodies(self) -> int:
        ...

    def body_names(self) -> list[str]:
        ...

    def ee_pos(self) -> np.ndarray:
        ...

    def body_xpos(self, body_id: int) -> np.ndarray:
        ...

    def ee_velocity(self) -> np.ndarray:
        ...

    def get_state(self) -> np.ndarray:
        ...

    def set_state(self, flat_state: np.ndarray) -> None:
        ...

    def forward(self) -> None:
        ...

    def step(self, action: Any) -> tuple[dict, float, bool, dict]:
        ...

    def reset(self) -> dict:
        ...

    def contact_forces(self) -> np.ndarray:
        ...


class RoboGymEnv:
    """A backend plus the task description and augmentation record."""

    def __init__(self, backend: SimBackend, task_description: str = "",
                 augmentation: dict | None = None, dt: float = 0.05):
        self.backend = backend
        self.task_description = task_description
        self.augmentation = augmentation or {}
        self.dt = dt

    def reset(self) -> dict:
        return self.backend.reset()

    def step(self, action: Any) -> tuple[dict, float, bool, dict]:
        return self.backend.step(action)

    @property
    def sim(self) -> SimBackend:
        return self.backend
