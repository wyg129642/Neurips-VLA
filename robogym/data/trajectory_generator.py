"""Automated expert-trajectory generation (paper §4.1).

The §4.1 pipeline samples 500 successful trajectories per task across the
50-task suite (25k total), balanced by logical domain. Each demo comes from
the per-task algorithmic oracle, mapped to physical waypoints, then smoothed
(cosine / B-spline) for jerk-free end-effector velocities. Rollouts that
fail the success predicate are discarded.

This module realises that pipeline on the physics reasoning sims. See
:func:`generate_task_demos` for the per-task sampler and
:func:`generate_dataset` for the suite-level streaming generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..tasks.registry import build_reasoning_suite

@dataclass
class GenConfig:
    demos_per_task: int = 500          # paper: 500 successful / task
    suite_seed: int = 0
    max_attempts_factor: int = 4       # resample budget to hit the quota
    max_steps: int = 600
    jitter: float = 0.012              # demo-to-demo diversity (avoid fixed script)
    require_success: bool = True       # keep only solved demonstrations
    balance_by: str = "category"       # logical-domain balancing key

@dataclass
class Demo:
    task_id: int
    task_name: str
    family: str
    category: str
    description: str
    actions: np.ndarray               # (T, 7) B-spline-smoothed EE deltas
    states: np.ndarray                # (T+?, S) flat sim states (demo_i/states)
    success: bool
    seed: int

def _rollout_states(sim, actions: np.ndarray, init_state: np.ndarray
                    ) -> tuple[np.ndarray, bool]:
    """Replay the oracle actions, recording the flat state at each step and
    whether the success predicate fired (used for success-filtering)."""
    sim.set_state(init_state)
    sim.forward()
    states = [sim.get_state().copy()]
    ok = False
    for a in actions:
        _, _, _, info = sim.step(np.asarray(a, float).tolist())
        states.append(sim.get_state().copy())
        if info.get("success"):
            ok = True
            break
    return np.asarray(states, float), ok

def generate_task_demos(task, cfg: GenConfig) -> list[Demo]:
    """Sample ``cfg.demos_per_task`` successful, randomised demos for one
    reasoning task (test-time randomization per demo, §3.6)."""
    out: list[Demo] = []
    budget = cfg.demos_per_task * cfg.max_attempts_factor
    attempt = 0
    while len(out) < cfg.demos_per_task and attempt < budget:
        seed = cfg.suite_seed * 100003 + task.task_id * 911 + attempt
        attempt += 1
        task.randomize(seed)
        sim = task.make_sim()
        actions, init_state = sim.generate_expert_demo(seed=seed,
                                                       jitter=cfg.jitter)
        if len(actions) == 0 or len(actions) > cfg.max_steps:
            continue
        states, ok = _rollout_states(sim, actions, init_state)
        if cfg.require_success and not ok:
            continue
        out.append(Demo(
            task_id=task.task_id, task_name=task.name, family=task.family,
            category=task.category, description=task.description,
            actions=np.asarray(actions, float), states=states,
            success=ok, seed=seed))
    return out

@dataclass
class GenReport:
    total_demos: int = 0
    per_task: dict = field(default_factory=dict)
    per_category: dict = field(default_factory=dict)
    shortfall: dict = field(default_factory=dict)

def generate_dataset(cfg: GenConfig, on_task=None):
    """Generate the full balanced expert dataset across the 50-task suite.

    Yields ``(task, demos)`` so callers can stream straight to disk (the 25k
    set is large). Returns a :class:`GenReport` of quotas met / shortfalls.
    """
    suite = build_reasoning_suite(seed=cfg.suite_seed)
    report = GenReport()
    for task in suite:
        demos = generate_task_demos(task, cfg)
        report.total_demos += len(demos)
        report.per_task[task.name] = len(demos)
        report.per_category[task.category] = (
            report.per_category.get(task.category, 0) + len(demos))
        if len(demos) < cfg.demos_per_task:
            report.shortfall[task.name] = cfg.demos_per_task - len(demos)
        if on_task is not None:
            on_task(task, demos)
        yield task, demos, report
