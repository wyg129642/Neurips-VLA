"""Evaluation runners."""

from .run_eval import EvalConfig, run_episode, run_suite, run_task

__all__ = ["EvalConfig", "run_suite", "run_task", "run_episode"]


def __getattr__(name):
    if name in ("Pi05EvalConfig", "eval_pi05"):
        from . import run_libero_eval_pi05
        return getattr(run_libero_eval_pi05, name)
    raise AttributeError(name)
