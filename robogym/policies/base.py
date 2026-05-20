"""Policy protocol used by the unified runner."""

from __future__ import annotations

from typing import Any

import numpy as np


class Policy:
    """Subclasses implement :meth:`act` and may override :meth:`reset`."""

    name: str = "policy"

    def reset(self, *, expert_actions: np.ndarray | None = None,
              expert_init_state: np.ndarray | None = None,
              task_description: str = "", **kw: Any) -> None:
        pass

    def act(self, observation: dict, task_description: str) -> np.ndarray:
        """Return an action chunk of shape ``(H, A)`` (or ``(A,)`` for H=1)."""
        raise NotImplementedError
