"""pi0 / pi0.5 LIBERO eval with multi-dimensional metrics.

Combines the openpi ``examples/libero`` rollout pattern with
:class:`robogym.metrics.TrajectoryEvaluator` via :class:`LiberoBackend`,
so pi0 / pi0.5 emit the same CSV schema as the OpenVLA-OFT runner.
Requires the full simulation stack and a running openpi policy server.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from ..metrics import (
    TrajectoryEvaluator,
    append_suite_row,
    init_summary_csvs,
    summarize_task,
    write_global_row,
    write_task_details,
)


@dataclasses.dataclass
class Pi05EvalConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    task_suite_name: str = "libero_object"
    datasets_root: str = "./LIBERO/datasets"
    num_trials_per_task: int = 10
    num_steps_wait: int = 10
    replan_steps: int = 5
    resize: int = 224
    results_dir: str = "./results/pi05"
    run_id_note: str | None = None
    seed: int = 7
    model_name: str = "pi05"


def _quat2axisangle(quat):
    import math

    q = np.asarray(quat, float)
    den = math.sqrt(max(1.0 - q[3] * q[3], 0.0))
    if den < 1e-8:
        return np.zeros(3)
    return (q[:3] / den) * (2.0 * math.acos(np.clip(q[3], -1.0, 1.0)))


def _prep_obs(obs, resize):
    import image_tools

    img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(obs["agentview_image"][::-1, ::-1],
                                    resize, resize))
    wrist = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(
            obs["robot0_eye_in_hand_image"][::-1, ::-1], resize, resize))
    state = np.concatenate([
        obs["robot0_eef_pos"],
        _quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"]])
    return {"observation/image": img, "observation/wrist_image": wrist,
            "observation/state": state}


def run_episode(cfg, env, backend, client, task_description,
                expert_actions, expert_init_state):
    from libero.libero import benchmark  # noqa: F401

    env.reset()
    obs = backend.env.set_init_state(expert_init_state)
    evaluator = TrajectoryEvaluator(backend, expert_actions,
                                    expert_init_state)

    action_plan = []
    t, success, max_steps = 0, False, 520
    while t < max_steps + cfg.num_steps_wait:
        if t < cfg.num_steps_wait:
            obs, _, done, info = env.step(
                [0, 0, 0, 0, 0, 0, -1])
            evaluator.update(np.zeros(7), info, step_idx=t)
            t += 1
            continue
        if not action_plan:
            element = _prep_obs(obs, cfg.resize)
            element["prompt"] = str(task_description)
            chunk = np.asarray(client.infer(element)["actions"], float)
            action_plan = list(chunk[: cfg.replan_steps])
        action = action_plan.pop(0)
        obs, _, done, info = env.step(action.tolist())
        evaluator.update(np.asarray(action, float), info, step_idx=t)
        if info.get("success") or done:
            success = bool(info.get("success", done))
            break
        t += 1
    return success, evaluator.calculate_score()


def eval_pi05(cfg: Pi05EvalConfig) -> dict:
    from libero.libero import benchmark
    from openpi_client import websocket_client_policy

    from ..envs.libero_env import LiberoBackend, make_libero_env

    client = websocket_client_policy.WebsocketClientPolicy(
        host=cfg.host, port=cfg.port)
    suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    names = suite.get_task_names()

    run_id = f"EVAL-CUSTOM-{cfg.task_suite_name}-{cfg.model_name}"
    if cfg.run_id_note:
        run_id += f"--{cfg.run_id_note}"
    out_dir = Path(cfg.results_dir) / run_id
    global_csv, suite_csv = init_summary_csvs(out_dir)

    import h5py

    per_task = []
    for task_id in range(suite.n_tasks):
        env, desc = make_libero_env(suite.get_task(task_id), "openpi",
                                    cfg.resize)
        backend = LiberoBackend(env)
        h5 = Path(cfg.datasets_root) / cfg.task_suite_name / \
            f"{names[task_id]}_demo.hdf5"
        rows = []
        with h5py.File(h5, "r") as f:
            keys = sorted(f["data"].keys(),
                          key=lambda x: int(x.split("_")[1]))
            for di in range(min(cfg.num_trials_per_task, len(keys))):
                k = keys[di]
                ea = f["data"][k]["actions"][:]
                eis = f["data"][k]["states"][0]
                ok, scores = run_episode(cfg, env, backend, client, desc,
                                         ea, eis)
                rows.append({"task_id": task_id, "task_name": names[task_id],
                             "task_description": desc, "demo_idx": di,
                             "success": int(ok), **scores})
        write_task_details(out_dir / cfg.task_suite_name
                           / f"task_{task_id:03d}_{names[task_id]}", rows)
        ts = summarize_task(cfg.task_suite_name, task_id, names[task_id],
                            desc, rows)
        per_task.append(ts)
        append_suite_row(suite_csv, ts)
    return write_global_row(global_csv, cfg.task_suite_name, per_task)


if __name__ == "__main__":
    import draccus

    eval_pi05(draccus.parse(Pi05EvalConfig))
