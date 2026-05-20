"""Policy protocol.

Generalises the OpenVLA-OFT runner's
``get_action(...) -> action_chunk`` + open-loop action-queue pattern
(``run_libero_eval_custom_metrics.py``) into a model-agnostic interface so the
eight VLA models compared in the paper (OpenVLA-OFT, π0, π0.5, GR00T-N1.6,
RDT, X-VLA, DM0, GigaBrain0.1, LingBot-VLA) all plug into one runner.
"""

from __future__ import annotations

from typing import Any

import numpy as np

class Policy:
    """Base policy. Subclasses implement :meth:`act` (and optionally
    :meth:`reset` to receive per-episode context such as the expert demo)."""

    #: human-readable model name used in result paths / tables
    name: str = "policy"

    def reset(self, *, expert_actions: np.ndarray | None = None,
              expert_init_state: np.ndarray | None = None,
              task_description: str = "", **kw: Any) -> None:
        """Called once at the start of each episode."""

    def act(self, observation: dict, task_description: str) -> np.ndarray:
        """Return an action *chunk*: shape ``(H, A)`` (or ``(A,)`` for H=1).

        The runner consumes it through an open-loop queue, matching the
        ``num_open_loop_steps`` behaviour.
        """
        raise NotImplementedError
