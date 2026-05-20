"""Unified RoboGym evaluation runner (backend / policy / scorer agnostic).

a model-agnostic re-implementation of the
``run_libero_eval_custom_metrics.py`` control flow (per-episode env
reset to the expert init state -> open-loop action queue -> per-step
``evaluator.update`` -> ``calculate_score`` -> task/suite/global CSV
schema). The original is preserved at
:mod:`robogym.runners.run_libero_eval_custom_metrics`.

Runs end-to-end on the synthetic backend with the oracle/mock policies, so the
whole pipeline is verifiable with no GPU / weights / mujoco.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..metrics import (
    TrajectoryEvaluator,
    append_suite_row,
    init_summary_csvs,
    score_episode,
    summarize_task,
    write_global_row,
    write_task_details,
)
from .suites import SyntheticSuite

@dataclass
class EvalConfig:
    backend: str = "synthetic"          # synthetic | libero
    scorer: str = "streaming"           # streaming | paper
    task_suite_name: str = "libero_object"
    num_tasks: int | None = None        # None == all
    num_trials_per_task: int = 5
    num_steps_wait: int = 10
    num_open_loop_steps: int = 8
    max_steps: int = 200
    expert_jitter: float = 0.0          # synth: jitter for demo diversity
    seed: int = 7
    results_dir: str = "./results"
    run_id_note: str | None = None
    datasets_root: str = ""             # libero HDF5 root
    dt: float = 0.05
    debug: bool = False

# observation shim (synthetic backend -> policy-facing obs dict)
def _policy_obs(raw_obs: dict) -> dict:
    """Map a backend observation dict to the policy-facing schema used by the
    OpenVLA-OFT runner (``full_image`` / ``wrist_image`` / ``state``). The
    synthetic backend has no camera, so image tensors are returned as zero
    arrays; the oracle/mock policies are state/replay driven and ignore them."""
    state = raw_obs.get("robot0_eef_pos", np.zeros(3))
    grip = raw_obs.get("robot0_gripper_qpos", np.zeros(2))
    return {
        "full_image": raw_obs.get("full_image",
                                  np.zeros((224, 224, 3), np.uint8)),
        "wrist_image": raw_obs.get("wrist_image",
                                   np.zeros((224, 224, 3), np.uint8)),
        "state": np.concatenate([np.asarray(state).ravel(),
                                 np.asarray(grip).ravel()]),
    }

def run_episode(cfg: EvalConfig, env, policy, expert_actions,
                expert_init_state) -> dict:
    """One rollout with the open-loop queue + streaming evaluator."""
    env.reset()
    env.backend.set_state(expert_init_state)
    env.backend.forward()

    evaluator = None
    record = {"ee": [], "speed": [], "force": []}
    if expert_actions is not None and expert_init_state is not None:
        evaluator = TrajectoryEvaluator(
            env.backend, expert_actions, expert_init_state, dt=cfg.dt,
            debug_mode=cfg.debug)

    policy.reset(expert_actions=expert_actions,
                 expert_init_state=expert_init_state,
                 task_description=env.task_description)

    queue: deque = deque(maxlen=cfg.num_open_loop_steps)
    obs = env.backend.reset()
    env.backend.set_state(expert_init_state)
    env.backend.forward()

    t, success = 0, False
    prev_ee = env.backend.ee_pos().copy()
    total_steps = cfg.max_steps + cfg.num_steps_wait
    while t < total_steps:
        if t < cfg.num_steps_wait:                       # settle 
            a = np.zeros(7)
            obs, _, done, info = env.step(a.tolist())
            if evaluator is not None:
                evaluator.update(a, info, step_idx=t)
            t += 1
            continue

        if not queue:
            chunk = np.atleast_2d(policy.act(_policy_obs(obs),
                                             env.task_description))
            queue.extend(list(chunk))
        action = np.asarray(queue.popleft(), dtype=float)

        obs, _, done, info = env.step(action.tolist())
        ee = env.backend.ee_pos()
        record["ee"].append(ee.copy())
        record["speed"].append(np.linalg.norm(ee - prev_ee) / cfg.dt)
        record["force"].append(float(np.max(np.linalg.norm(
            env.backend.contact_forces(), axis=1))))
        prev_ee = ee.copy()

        if evaluator is not None:
            evaluator.update(action, info, step_idx=t)
        if info.get("success") or done:
            success = info.get("success", True)
            break
        t += 1

    # scoring
    if cfg.scorer == "paper" and evaluator is not None:
        ee_path = np.asarray(record["ee"]) if record["ee"] else \
            evaluator.expert_ee_path
        # phases: reach segment + per moved-object segment (expert paths)
        phases = [evaluator.expert_ee_path[:evaluator.reach_cutoff_idx + 1]]
        phases += [o["path"] for o in evaluator.tracked_objects]
        scores = score_episode(
            ee_path, phases, None,
            np.asarray(record["speed"]) if record["speed"] else np.zeros(2),
            np.asarray(record["force"]) if record["force"] else np.zeros(1),
            expert_total_len=float(evaluator.ee_total_len),
            t_exec=float(len(record["ee"]) * cfg.dt),
            v_expert=float(evaluator.ee_total_len
                           / max(len(expert_actions) * cfg.dt, 1e-6)),
            fs=1.0 / cfg.dt, success=success)
    else:
        scores = (evaluator.calculate_score() if evaluator is not None
                  else TrajectoryEvaluator._empty_score())
    return {"success": int(success), **scores}

def run_task(cfg: EvalConfig, suite, task_id: int, policy,
             out_dir: Path) -> dict | None:
    spec = suite.task_spec(task_id)
    env = suite.make_env(task_id)
    rows = []
    for demo_idx in range(cfg.num_trials_per_task):
        ea, eis = suite.expert_demo(env, demo_idx, jitter=cfg.expert_jitter)
        res = run_episode(cfg, env, policy, ea, eis)
        rows.append({"task_id": task_id, "task_name": spec.task_name,
                     "task_description": spec.task_description,
                     "demo_idx": demo_idx, **res})
    if not rows:
        return None
    task_dir = out_dir / cfg.task_suite_name / \
        f"task_{task_id:03d}_{spec.task_name}"
    write_task_details(task_dir, rows)
    return summarize_task(cfg.task_suite_name, task_id, spec.task_name,
                          spec.task_description, rows)

def run_suite(cfg: EvalConfig, policy) -> dict:
    """Full suite eval -> CSV tree. Returns the global summary row."""
    np.random.seed(cfg.seed)
    if cfg.backend == "synthetic":
        suite = SyntheticSuite(cfg.task_suite_name, dt=cfg.dt, seed=cfg.seed)
    elif cfg.backend == "reasoning":
        from .suites import ReasoningRunnerSuite
        suite = ReasoningRunnerSuite(cfg.task_suite_name, dt=cfg.dt,
                                     seed=cfg.seed)
    elif cfg.backend == "libero":
        from .suites import LiberoSuite
        suite = LiberoSuite(cfg.task_suite_name, cfg.datasets_root, dt=cfg.dt)
    else:
        raise ValueError(f"unknown backend {cfg.backend!r}")

    run_id = f"EVAL-CUSTOM-{cfg.task_suite_name}-{policy.name}"
    if cfg.run_id_note:
        run_id += f"--{cfg.run_id_note}"
    out_dir = Path(cfg.results_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    global_csv, suite_csv = init_summary_csvs(out_dir)
    with open(out_dir / "config.json", "w") as f:
        json.dump({**asdict(cfg), "policy": policy.name}, f, indent=2)

    n = suite.n_tasks if cfg.num_tasks is None else min(cfg.num_tasks,
                                                        suite.n_tasks)
    per_task = []
    for task_id in range(n):
        ts = run_task(cfg, suite, task_id, policy, out_dir)
        if ts is None:
            continue
        per_task.append(ts)
        append_suite_row(suite_csv, ts)
        print(f"[{policy.name}] task {task_id:02d} "
              f"{ts['task_name'][:42]:42s} "
              f"SR={ts['success_rate']:.2f} "
              f"Total={ts['avg_total_fwdbias']:6.2f} "
              f"Comp={ts['avg_completion']:6.2f}")
    grow = write_global_row(global_csv, cfg.task_suite_name, per_task)
    print(f"\n[{policy.name}] SUITE {cfg.task_suite_name}: "
          f"SR={grow.get('success_rate', 0):.3f} "
          f"Total={grow.get('avg_total_fwdbias', 0):.2f} | CSVs -> {out_dir}")
    return grow
