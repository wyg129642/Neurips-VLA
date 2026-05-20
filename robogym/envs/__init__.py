"""RoboGym environment backends.

- :class:`RoboGymEnv` and :class:`SimBackend`: the backend protocol. The
  streaming metric engine runs against this protocol unchanged on every
  backend.
- :class:`SyntheticPickPlaceSim`: a dependency-free sim runnable everywhere
  (used by the demo and tests).
- :class:`LiberoBackend` and :func:`build_augmented_env`: real LIBERO plus
  the Figure-1 task-augmentation wrapper (requires the full sim stack).
"""

from .base import RoboGymEnv, SimBackend
from .synthetic_env import SyntheticPickPlaceSim

__all__ = ["RoboGymEnv", "SimBackend", "SyntheticPickPlaceSim"]

try:  # optional; only importable with robosuite/mujoco/libero installed.
    from .libero_env import LiberoBackend, build_augmented_env, make_libero_env

    __all__ += ["LiberoBackend", "build_augmented_env", "make_libero_env"]
except ImportError:  # pragma: no cover - heavy-stack imports are optional
    pass
