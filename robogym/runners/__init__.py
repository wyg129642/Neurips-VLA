"""RoboGym evaluation runners.

- :func:`run_suite`, :func:`run_task`, :func:`run_episode` (from
  ``run_eval``): the unified, backend/policy/scorer-agnostic runner.
  Runnable with no GPU on the synthetic backend.
- ``run_libero_eval_custom_metrics``: the OpenVLA-OFT LIBERO eval runner
  with multi-dimensional trajectory metrics (full simulation stack).
- ``run_libero_eval_pi05``: the openpi pi0 / pi0.5 LIBERO eval runner with
  the same multi-dimensional metrics.
"""

from .run_eval import EvalConfig, run_episode, run_suite, run_task

__all__ = ["EvalConfig", "run_suite", "run_task", "run_episode"]

def __getattr__(name):
    if name in ("Pi05EvalConfig", "eval_pi05"):
        from . import run_libero_eval_pi05
        return getattr(run_libero_eval_pi05, name)
    raise AttributeError(name)
