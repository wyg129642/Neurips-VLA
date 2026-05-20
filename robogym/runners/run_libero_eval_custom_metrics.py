"""
run_libero_eval_custom_metrics.py

OpenVLA LIBERO eval + custom trajectory metrics (TrajectoryEvaluator)
- Uses expert demos (.hdf5) to reset env init_state per episode (demo_0..demo_N)
- Runs OpenVLA policy rollout
- Computes custom metrics aligned with pi0.5 evaluator_trajectory.py
- Saves videos + task/suite/global CSV summaries
"""

import csv
import datetime
import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Tuple

import draccus
import h5py
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb

# Ensure the interpreter can find experiments.robot when this file is run
# as a script rather than as a module from the repo root.
sys.path.append("../..")

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

# Your custom evaluator (copy file into PYTHONPATH or same dir)
from experiments.robot.libero.evaluator_trajectory import TrajectoryEvaluator

# Task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"

TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,
    TaskSuite.LIBERO_OBJECT: 280,
    TaskSuite.LIBERO_GOAL: 300,
    TaskSuite.LIBERO_10: 520,
    TaskSuite.LIBERO_90: 400,
}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

def log_message(message: str, log_file=None):
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()

# Expert demo loader
class ExpertDataLoader:
    """
    Loads expert demos from:
        {datasets_root}/{suite_name}/{task_name}_demo.hdf5
    Expects structure:
        f["data"]["demo_0"]["actions"][:]    -> (T, 7) actions
        f["data"]["demo_0"]["states"][0]     -> flattened mujoco state for init reset
    """

    def __init__(self, task_name: str, suite_name: str, datasets_root: str):
        self.hdf5_path = os.path.join(datasets_root, suite_name, f"{task_name}_demo.hdf5")
        self.valid = os.path.exists(self.hdf5_path)
        self.f = None
        self.demo_keys: List[str] = []

        if not self.valid:
            return

        try:
            self.f = h5py.File(self.hdf5_path, "r")
            self.demo_keys = list(self.f["data"].keys())
            self.demo_keys.sort(key=lambda x: int(x.split("_")[1]))  # demo_0, demo_1, ...
        except Exception as e:
            logger.warning(f"Failed to open HDF5 {self.hdf5_path}: {e}")
            self.valid = False
            self.f = None
            self.demo_keys = []

    def __len__(self):
        return len(self.demo_keys)

    def get_demo(self, index: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if (not self.valid) or (index >= len(self.demo_keys)):
            return None, None
        demo_key = self.demo_keys[index]
        try:
            actions = self.f["data"][demo_key]["actions"][:]
            init_state = self.f["data"][demo_key]["states"][0]
            return actions, init_state
        except Exception as e:
            logger.warning(f"Failed to read demo {demo_key} from {self.hdf5_path}: {e}")
            return None, None

    def close(self):
        if self.f:
            self.f.close()
            self.f = None

# Config
@dataclass
class GenerateConfig:
    # Model
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = 8
    lora_rank: int = 32
    unnorm_key: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # LIBERO
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL
    num_steps_wait: int = 10
    num_trials_per_task: int = 10  # interpret as "use first N expert demos"
    env_img_res: int = 256

    # Custom metrics / expert demos
    datasets_root: str = "/inspire/hdd/project/wuliqifa/czxs25210147/LIBERO/datasets"
    # if True, reset env with expert_init_state (recommended for your custom eval)
    use_expert_init_state: bool = False
    # if True, evaluator uses expert_actions + expert_init_state each episode
    use_custom_metrics: bool = True

    # Utils / logging
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    results_dir: str = "./experiments/custom_metrics"  # CSV + videos dir root
    use_wandb: bool = False
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"
    seed: int = 7

def validate_config(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None and str(cfg.pretrained_checkpoint) != "", "pretrained_checkpoint required"
    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expect center_crop=True because model was trained with image aug"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"
    if cfg.use_expert_init_state:
        assert os.path.exists(cfg.datasets_root), f"datasets_root not found: {cfg.datasets_root}"

def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    unnorm_key = cfg.task_suite_name
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in model.norm_stats"
    cfg.unnorm_key = unnorm_key

def initialize_model(cfg: GenerateConfig):
    model = get_model(cfg)

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)

    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor

def setup_logging_and_dirs(cfg: GenerateConfig):
    run_id = f"EVAL-CUSTOM-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note:
        run_id += f"--{cfg.run_id_note}"

    os.makedirs(cfg.local_log_dir, exist_ok=True)
    log_path = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(log_path, "w")
    log_message(f"Logging to: {log_path}", log_file)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.results_dir) / f"{run_id}-{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # W&B
    if cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)

    return log_file, run_id, out_dir

# Observation / action helpers
def prepare_observation(obs, resize_size):
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }
    return observation, img

def process_action(action, model_family):
    action = normalize_gripper_action(action, binarize=True)
    if model_family == "openvla":
        action = invert_gripper_action(action)
    return action

def set_env_init_state(env, init_state: np.ndarray):
    """Set the env initial state from a flattened MuJoCo state.

    Some env wrappers provide ``env.set_init_state``; otherwise we fall
    back to ``sim.set_state_from_flattened``.
    """
    try:
        obs = env.set_init_state(init_state)
        return obs
    except (AttributeError, TypeError, ValueError):
        # Wrapper does not expose set_init_state, or the flat-state shape
        # does not match; fall through to the MuJoCo sim path below.
        pass

    # Fallback to MuJoCo sim.
    try:
        env.sim.set_state_from_flattened(init_state)
        env.sim.forward()
        return env.get_observation()
    except Exception as e:
        raise RuntimeError(f"Failed to set init state via both env.set_init_state and sim.set_state_from_flattened: {e}")

# Episode + Task evaluation
def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    expert_actions: Optional[np.ndarray] = None,
    expert_init_state: Optional[np.ndarray] = None,
    log_file=None,
):
    """
    Runs one rollout, optionally computing custom metrics with TrajectoryEvaluator.
    Returns: success(bool), replay_images(list), scores(dict or None)
    """
    env.reset()

    # set init state (expert-aligned)
    if cfg.use_expert_init_state and (expert_init_state is not None):
        obs = set_env_init_state(env, expert_init_state)
    else:
        obs = env.get_observation()

    evaluator = None
    if cfg.use_custom_metrics and (expert_actions is not None) and (expert_init_state is not None):
        evaluator = TrajectoryEvaluator(env, expert_actions, expert_init_state)

    # action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        log_message(
            f"WARNING: num_open_loop_steps ({cfg.num_open_loop_steps}) != NUM_ACTIONS_CHUNK ({NUM_ACTIONS_CHUNK}). "
            f"Recommend executing full chunk for best speed/success.",
            log_file,
        )
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    success = False

    try:
        while t < max_steps + cfg.num_steps_wait:
            # stabilize steps
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                if evaluator is not None:
                    evaluator.update(get_libero_dummy_action(cfg.model_family), info, step_idx=t)
                t += 1
                continue

            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            if len(action_queue) == 0:
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions)

            action = action_queue.popleft()
            action = process_action(action, cfg.model_family)

            obs, reward, done, info = env.step(action.tolist())

            if evaluator is not None:
                evaluator.update(action, info, step_idx=t)

            if done:
                success = True
                break

            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    scores = evaluator.calculate_score() if evaluator is not None else None
    return success, replay_images, scores

def _mean_metric(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [r.get(key, 0.0) for r in rows if r.get(key, None) is not None]
    return float(np.mean(vals)) if len(vals) > 0 else 0.0

def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    out_dir: Path = Path("./"),
    log_file=None,
):
    task = task_suite.get_task(task_id)
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    task_names = task_suite.get_task_names()
    task_name = task_names[task_id]

    # Official init states (used by LIBERO / OpenVLA baseline eval)
    initial_states = task_suite.get_task_init_states(task_id)
    num_official_inits = len(initial_states)

    # Expert demos (optional; needed if you want custom metrics aligned to expert trajectories)
    loader = ExpertDataLoader(task_name=task_name, suite_name=cfg.task_suite_name, datasets_root=cfg.datasets_root)
    has_hdf5 = loader.valid

    if cfg.use_expert_init_state:
        if not has_hdf5:
            log_message(f"[SKIP] HDF5 not found for task {task_name} at {loader.hdf5_path}", log_file)
            return total_episodes, total_successes, None
        num_trials_available = len(loader)
        source_note = f"HDF5 demos (expert init_state) from {loader.hdf5_path}"
    else:
        num_trials_available = num_official_inits
        source_note = "official task_suite init_states"

    num_demos_to_run = min(cfg.num_trials_per_task, num_trials_available)
    log_message(f"\nTask {task_id}: {task_description}", log_file)
    log_message(f"Init-state source: {source_note}", log_file)
    log_message(f"Running episodes: 0..{num_demos_to_run-1}", log_file)

    episode_rows: List[Dict[str, Any]] = []

    for demo_idx in tqdm.tqdm(range(num_demos_to_run)):
        expert_actions, expert_init_state = None, None
        init_state_to_use = None

        if cfg.use_expert_init_state:
            # Use HDF5 init_state (aligned with expert trajectory)
            expert_actions, expert_init_state = loader.get_demo(demo_idx)
            if expert_actions is None or expert_init_state is None:
                log_message(f"[WARN] demo_{demo_idx} read failed, skipping", log_file)
                continue
            init_state_to_use = expert_init_state
        else:
            # Use official init_state (parity with official eval)
            init_state_to_use = initial_states[demo_idx]

            # Optional: still load expert_actions for metrics.
            # WARNING: if expert_init_state != init_state_to_use, TrajectoryEvaluator alignment may be invalid.
            if cfg.use_custom_metrics and has_hdf5:
                expert_actions, expert_init_state = loader.get_demo(min(demo_idx, len(loader)-1))
                # Best practice: disable custom metrics when not using expert init_state
                # because evaluator assumes expert_init_state alignment.
                # (You can override if you really want, but expect weird scores.)
                # if expert_init_state is None or not np.allclose(expert_init_state, init_state_to_use):
                #     expert_actions, expert_init_state = None, None

        log_message(f"Starting episode {demo_idx}...", log_file)

        # Pass init_state_to_use through run_episode's expert-init-state path
        # by temporarily enabling cfg.use_expert_init_state for this call.
        orig_use_expert = cfg.use_expert_init_state
        if not orig_use_expert:
            cfg.use_expert_init_state = True  # reuse existing reset path
            expert_init_state_for_reset = init_state_to_use
        else:
            expert_init_state_for_reset = init_state_to_use

        success, replay_images, scores = run_episode(
            cfg=cfg,
            env=env,
            task_description=task_description,
            model=model,
            resize_size=resize_size,
            processor=processor,
            action_head=action_head,
            proprio_projector=proprio_projector,
            noisy_action_projector=noisy_action_projector,
            expert_actions=expert_actions if cfg.use_custom_metrics else None,
            expert_init_state=expert_init_state_for_reset,
            log_file=log_file,
        )

        # restore flag
        cfg.use_expert_init_state = orig_use_expert

        total_episodes += 1
        if success:
            total_successes += 1

        save_rollout_video(
            replay_images,
            total_episodes,
            success=success,
            task_description=task_description,
            log_file=log_file,
        )

        row = {
            "task_id": task_id,
            "task_name": task_name,
            "task_description": task_description,
            "demo_idx": demo_idx,
            "success": int(success),
        }
        if scores is not None:
            row.update(scores)
        episode_rows.append(row)

        log_message(f"Success: {success}", log_file)

    if has_hdf5:
        loader.close()

    if len(episode_rows) == 0:
        return total_episodes, total_successes, None

    # below unchanged: write CSV, aggregate metrics
    task_dir = out_dir / cfg.task_suite_name / f"task_{task_id:03d}_{task_name}"
    task_dir.mkdir(parents=True, exist_ok=True)
    task_details_csv = task_dir / "task_details.csv"

    metric_cols = [
        "total_fwdbias",
        "total_hybrid",
        "completion",
        "space_eff",
        "time_eff",
        "smoothness",
        "safety",
        "raw_dev",
        "raw_space_val",
        "raw_time_val",
        "raw_fft_val",
        "dropped",
    ]
    header = ["task_id", "task_name", "task_description", "demo_idx", "success"] + metric_cols

    with open(task_details_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in episode_rows:
            for k in metric_cols:
                if k not in r:
                    r[k] = 0.0 if k != "dropped" else False
            w.writerow(r)

    task_summary = {
        "task_id": task_id,
        "task_name": task_name,
        "task_description": task_description,
        "num_episodes": len(episode_rows),
        "success_rate": float(np.mean([r["success"] for r in episode_rows])),
    }
    for k in metric_cols:
        task_summary[f"avg_{k}"] = _mean_metric(episode_rows, k)

    if cfg.use_wandb:
        wandb.log({f"success_rate/{task_description}": task_summary["success_rate"]})

    return total_episodes, total_successes, task_summary

@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)

    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)

    log_file, run_id, out_dir = setup_logging_and_dirs(cfg)
    log_message(f"Results dir: {out_dir}", log_file)

    # Prepare global CSV paths
    global_csv = out_dir / "global_summary.csv"
    suite_csv = out_dir / "suite_summary.csv"

    # Header definitions
    suite_header = [
        "suite_name",
        "num_tasks",
        "num_episodes",
        "success_rate",
        "avg_total_fwdbias",
        "avg_completion",
        "avg_space_eff",
        "avg_time_eff",
        "avg_smoothness",
        "avg_safety",
        "avg_raw_dev",
        "avg_dropped_rate",
    ]
    task_header = [
        "suite_name",
        "task_id",
        "task_name",
        "num_episodes",
        "success_rate",
        "avg_total_fwdbias",
        "avg_completion",
        "avg_space_eff",
        "avg_time_eff",
        "avg_smoothness",
        "avg_safety",
        "avg_raw_dev",
        "avg_dropped",
    ]

    with open(global_csv, "w", newline="") as f:
        csv.writer(f).writerow(suite_header)
    with open(suite_csv, "w", newline="") as f:
        csv.writer(f).writerow(task_header)

    # Init task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name} (num_tasks={num_tasks})", log_file)

    total_episodes, total_successes = 0, 0
    per_task_summaries: List[Dict[str, Any]] = []

    for task_id in tqdm.tqdm(range(num_tasks)):
        total_episodes, total_successes, task_summary = run_task(
            cfg=cfg,
            task_suite=task_suite,
            task_id=task_id,
            model=model,
            resize_size=resize_size,
            processor=processor,
            action_head=action_head,
            proprio_projector=proprio_projector,
            noisy_action_projector=noisy_action_projector,
            total_episodes=total_episodes,
            total_successes=total_successes,
            out_dir=out_dir,
            log_file=log_file,
        )

        if task_summary is None:
            continue

        per_task_summaries.append(task_summary)

        # Append to suite_summary.csv (per task)
        with open(suite_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    cfg.task_suite_name,
                    task_summary["task_id"],
                    task_summary["task_name"],
                    task_summary["num_episodes"],
                    round(task_summary["success_rate"], 4),
                    round(task_summary["avg_total_fwdbias"], 2),
                    round(task_summary["avg_completion"], 2),
                    round(task_summary["avg_space_eff"], 2),
                    round(task_summary["avg_time_eff"], 2),
                    round(task_summary["avg_smoothness"], 2),
                    round(task_summary["avg_safety"], 2),
                    round(task_summary["avg_raw_dev"], 3),
                    round(task_summary["avg_dropped"], 4),
                ]
            )

    # Final aggregate over tasks
    if len(per_task_summaries) == 0:
        log_message("No tasks evaluated successfully. Check datasets_root / suite name / HDF5 paths.", log_file)
        return 0.0

    suite_num_episodes = int(np.sum([t["num_episodes"] for t in per_task_summaries]))
    suite_success_rate = float(np.mean([t["success_rate"] for t in per_task_summaries]))

    suite_row = {
        "suite_name": cfg.task_suite_name,
        "num_tasks": len(per_task_summaries),
        "num_episodes": suite_num_episodes,
        "success_rate": suite_success_rate,
        "avg_total_fwdbias": float(np.mean([t["avg_total_fwdbias"] for t in per_task_summaries])),
        "avg_completion": float(np.mean([t["avg_completion"] for t in per_task_summaries])),
        "avg_space_eff": float(np.mean([t["avg_space_eff"] for t in per_task_summaries])),
        "avg_time_eff": float(np.mean([t["avg_time_eff"] for t in per_task_summaries])),
        "avg_smoothness": float(np.mean([t["avg_smoothness"] for t in per_task_summaries])),
        "avg_safety": float(np.mean([t["avg_safety"] for t in per_task_summaries])),
        "avg_raw_dev": float(np.mean([t["avg_raw_dev"] for t in per_task_summaries])),
        "avg_dropped_rate": float(np.mean([t["avg_dropped"] for t in per_task_summaries])),
    }

    with open(global_csv, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                suite_row["suite_name"],
                suite_row["num_tasks"],
                suite_row["num_episodes"],
                round(suite_row["success_rate"], 4),
                round(suite_row["avg_total_fwdbias"], 2),
                round(suite_row["avg_completion"], 2),
                round(suite_row["avg_space_eff"], 2),
                round(suite_row["avg_time_eff"], 2),
                round(suite_row["avg_smoothness"], 2),
                round(suite_row["avg_safety"], 2),
                round(suite_row["avg_raw_dev"], 3),
                round(suite_row["avg_dropped_rate"], 4),
            ]
        )

    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)
    log_message(f"Suite avg total_fwdbias: {suite_row['avg_total_fwdbias']:.2f}", log_file)
    log_message(f"CSV saved: {global_csv} and {suite_csv}", log_file)

    if cfg.use_wandb:
        wandb.log({"success_rate/total": final_success_rate})
        wandb.save(str(global_csv))
        wandb.save(str(suite_csv))

    if log_file:
        log_file.close()

    return final_success_rate

if __name__ == "__main__":
    eval_libero()
