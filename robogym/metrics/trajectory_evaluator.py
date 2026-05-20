"""Streaming multi-dimensional evaluator.

Updates phase progress, spatial / temporal efficiency, SPARC smoothness,
and the tripartite safety penalty as actions arrive. The simulator is
accessed only through :class:`robogym.envs.base.SimBackend`, so the same
code runs on real LIBERO and on the synthetic backend.

The constructor accepts an optional ``augmentation`` record produced by
:class:`robogym.augmentation.AugmentationPipeline` that supplies the safe
workspace and the expert-derived force thresholds; without it, the
defaults from Sec. 3.5 are used.
"""

from __future__ import annotations

from collections import deque

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

from .safety import safety_score
from .sparc import sparc_score


class TrajectoryEvaluator:
    def __init__(self, backend, expert_actions, expert_init_state,
                 dt: float = 0.05, augmentation: dict | None = None,
                 debug_mode: bool = False):
        self.env = backend
        self.dt = dt
        self.debug_mode = debug_mode
        aug = augmentation or {}
        self.augmentation = aug
        self.gamma_expert = float(aug.get("gamma_expert", 1.0))
        self.f_sus_lim = float(aug.get("f_sus_lim", 40.0))
        self.f_peak_lim = float(aug.get("f_peak_lim", 60.0))
        ws = aug.get("workspace")
        self.workspace = (np.asarray(ws[0], float),
                          np.asarray(ws[1], float)) if ws else None
        self.phase_weights = aug.get("phase_weights")

        (self.expert_ee_path, self.tracked_objects, self.reach_weight,
         self.reach_cutoff_idx) = self._extract_expert_multi_paths(
            expert_actions, expert_init_state)

        self.expert_ee_arc_lengths = self._calc_arc_lengths(self.expert_ee_path)
        self.ee_total_len = self.expert_ee_arc_lengths[-1]
        self.expert_v = self.ee_total_len / max(len(expert_actions) * self.dt,
                                                 1e-6)

        self.current_step = 0
        self.current_focus_stage = -1
        self.last_ee_idx = 0
        self.last_obj_indices = {obj["id"]: 0 for obj in self.tracked_objects}

        self.max_ee_progress_dist = 0.0
        self.max_unified_progress_ratio = 0.0
        self.prev_unified_progress_ratio = 0.0
        self.obj_max_progress = {obj["id"]: 0.0 for obj in self.tracked_objects}

        self.initialized = False
        self.ee_path_offset = np.zeros(3)
        self.last_ee_pos = None
        self.total_actual_path = 0.0

        self.velocity_history: list[float] = []
        self.velocity_window: deque = deque(maxlen=20)
        self.ee_positions: list[np.ndarray] = []

        self.metrics = {
            "time": [],
            "ee_progress": [],
            "obj_progress_weighted": [],
            "unified_progress": [],
            "deviation": [],
            "space_eff_cum": [], "time_eff_cum": [],
            "force": [], "success": False,
        }

    def _extract_expert_multi_paths(self, actions, init_state):
        if self.debug_mode:
            print("\n[DEBUG] Extracting reference paths from expert demo")

        old_state = self.env.get_state()
        self.env.set_state(init_state)
        self.env.forward()

        obj_names = self.env.body_names()
        exclude_keywords = [
            "robot", "world", "floor", "mount", "gripper", "site",
            "finger", "left_pad", "right_pad", "base", "camera", "visual",
            "collision", "link", "inertial", "light", "lamp", "target", "ref",
        ]

        valid_obj_ids = [
            i for i, name in enumerate(obj_names)
            if name and not any(x in name for x in exclude_keywords)
        ]

        initial_poses = {i: self.env.body_xpos(i).copy() for i in valid_obj_ids}
        ee_path = [self.env.ee_pos().copy()]
        obj_paths_dict = {i: [initial_poses[i]] for i in valid_obj_ids}

        for action in actions:
            self.env.step(action)
            ee_path.append(self.env.ee_pos().copy())
            for i in valid_obj_ids:
                obj_paths_dict[i].append(self.env.body_xpos(i).copy())

        ee_path_np = np.array(ee_path)
        moving_objects = []
        first_move_step_global = len(actions)

        for i in valid_obj_ids:
            path = np.array(obj_paths_dict[i])
            displacements = np.linalg.norm(path - path[0], axis=1)
            max_displacement = np.max(displacements)

            if max_displacement <= 0.03:
                continue
            dists_to_ee = np.linalg.norm(path - ee_path_np, axis=1)
            if np.min(dists_to_ee) >= 0.25:
                continue

            start_indices = np.where(displacements > 0.01)[0]
            if len(start_indices) == 0:
                continue
            t_start_idx = start_indices[0]
            dist_to_final = np.linalg.norm(path - path[-1], axis=1)
            not_settled = np.where(dist_to_final > 0.01)[0]
            t_end_idx = not_settled[-1] if len(not_settled) > 0 else t_start_idx
            duration = (t_end_idx - t_start_idx) * self.dt
            if duration <= 1.0:
                continue

            diffs = np.linalg.norm(np.diff(path, axis=0), axis=1)
            total_len = np.sum(diffs)
            first_move_step_global = min(first_move_step_global, t_start_idx)
            arc_lens = self._calc_arc_lengths(path)
            moving_objects.append({
                "id": i,
                "name": obj_names[i],
                "path": path,
                "arc_lengths": arc_lens,
                "total_len": arc_lens[-1],
                "total_disp": total_len,
                "max_disp": max_displacement,
                "start_idx": t_start_idx,
                "duration": duration,
            })

        if self.debug_mode:
            print(f"[DEBUG] Tracked objects: {[o['name'] for o in moving_objects]}")

        ee_diffs = np.linalg.norm(np.diff(ee_path_np, axis=0), axis=1)
        ee_cum_dist = np.concatenate(([0], np.cumsum(ee_diffs)))
        reach_len = (ee_cum_dist[first_move_step_global]
                     if first_move_step_global < len(ee_cum_dist)
                     else ee_cum_dist[-1])

        if not moving_objects:
            w_reach = 1.0
            total_task_metric = reach_len
        else:
            total_obj_disp = sum(o["total_disp"] for o in moving_objects)
            total_task_metric = reach_len + total_obj_disp + 1e-6
            w_reach = reach_len / total_task_metric

        final_objects = []
        for o in moving_objects:
            o["weight"] = o["total_disp"] / (total_task_metric if moving_objects
                                             else 1.0)
            final_objects.append(o)
        final_objects.sort(key=lambda x: x["start_idx"])

        self.env.set_state(old_state)
        self.env.forward()
        return ee_path_np, final_objects, w_reach, first_move_step_global

    @staticmethod
    def _calc_arc_lengths(path):
        if len(path) < 2:
            return np.zeros(len(path))
        diffs = np.linalg.norm(np.diff(path, axis=0), axis=1)
        return np.concatenate(([0], np.cumsum(diffs)))

    def _windowed_argmin(self, target_pos, reference_path, last_idx,
                         window_size=20):
        start = max(0, last_idx - 5)
        end = min(len(reference_path), last_idx + window_size)
        if start >= end:
            return last_idx, np.linalg.norm(reference_path[last_idx] - target_pos)
        sub_path = reference_path[start:end]
        dists = np.linalg.norm(sub_path - target_pos, axis=1)
        local_min_idx = np.argmin(dists)
        min_dist = dists[local_min_idx]
        global_idx = start + local_min_idx
        if min_dist > 0.20:
            global_dists = np.linalg.norm(reference_path - target_pos, axis=1)
            global_min_idx = np.argmin(global_dists)
            global_min_dist = global_dists[global_min_idx]
            if global_min_dist < min_dist - 0.05:
                return global_min_idx, global_min_dist
        return global_idx, min_dist

    def update(self, action, info, step_idx: int = 0):
        self.current_step = step_idx
        curr_time = step_idx * self.dt
        ee_pos = self.env.ee_pos().copy()

        if not self.initialized:
            self.ee_path_offset = ee_pos - self.expert_ee_path[0]
            self.initialized = True
            self.last_ee_pos = ee_pos

        aligned_ee_path = self.expert_ee_path + self.ee_path_offset
        ee_idx, ee_min_dist = self._windowed_argmin(
            ee_pos, aligned_ee_path, self.last_ee_idx, window_size=20)
        self.last_ee_idx = ee_idx

        clamped_ee_idx = min(ee_idx, self.reach_cutoff_idx)
        current_ee_dist = self.expert_ee_arc_lengths[clamped_ee_idx]
        self.max_ee_progress_dist = max(self.max_ee_progress_dist,
                                        current_ee_dist)
        reach_len_limit = self.expert_ee_arc_lengths[self.reach_cutoff_idx] + 1e-6
        reach_ratio_local = min(1.0, self.max_ee_progress_dist / reach_len_limit)

        current_obj_ratios = []
        for obj in self.tracked_objects:
            curr_obj_pos = self.env.body_xpos(obj["id"]).copy()
            o_idx, _ = self._windowed_argmin(
                curr_obj_pos, obj["path"], self.last_obj_indices[obj["id"]],
                window_size=30)
            self.last_obj_indices[obj["id"]] = o_idx
            curr_o_dist = obj["arc_lengths"][o_idx]
            self.obj_max_progress[obj["id"]] = max(
                self.obj_max_progress[obj["id"]], curr_o_dist)
            current_obj_ratios.append(
                self.obj_max_progress[obj["id"]] / (obj["total_len"] + 1e-6))

        if self.current_focus_stage == -1 and len(self.tracked_objects) > 0:
            if reach_ratio_local > 0.95 or current_obj_ratios[0] > 0.05:
                self.current_focus_stage = 0

        while (0 <= self.current_focus_stage < len(self.tracked_objects)):
            curr_stage = self.current_focus_stage
            advance = current_obj_ratios[curr_stage] > 0.95
            nxt = curr_stage + 1
            if nxt < len(self.tracked_objects) and current_obj_ratios[nxt] > 0.05:
                advance = True
            if advance and nxt < len(self.tracked_objects):
                self.current_focus_stage += 1
            else:
                break

        total_obj_contribution = 0.0
        weighted_obj_prog_val = 0.0
        total_obj_weight = sum(o["weight"]
                               for o in self.tracked_objects) + 1e-9
        for idx, obj in enumerate(self.tracked_objects):
            if idx <= self.current_focus_stage:
                o_ratio = current_obj_ratios[idx]
                total_obj_contribution += o_ratio * obj["weight"]
                weighted_obj_prog_val += o_ratio * (obj["weight"]
                                                    / total_obj_weight)

        unified_ratio = ((self.reach_weight * reach_ratio_local)
                         + total_obj_contribution)
        self.max_unified_progress_ratio = max(self.max_unified_progress_ratio,
                                              unified_ratio)

        step_dist = np.linalg.norm(ee_pos - self.last_ee_pos)
        self.total_actual_path += step_dist
        vel = step_dist / self.dt
        self.velocity_history.append(vel)
        self.velocity_window.append(vel)
        self.ee_positions.append(ee_pos.copy())

        delta_prog = (self.max_unified_progress_ratio
                      - self.prev_unified_progress_ratio)
        self.prev_unified_progress_ratio = self.max_unified_progress_ratio

        if self.total_actual_path > 1e-4:
            s_eff_cum = ((self.max_unified_progress_ratio * self.ee_total_len)
                         / self.total_actual_path)
        else:
            s_eff_cum = 1.0

        if curr_time > 1e-4:
            t_eff_cum = ((self.max_unified_progress_ratio * self.ee_total_len)
                         / (curr_time * (self.expert_v + 1e-9)))
        else:
            t_eff_cum = 0.0

        self.metrics["time"].append(curr_time)
        self.metrics["ee_progress"].append(reach_ratio_local)
        self.metrics["obj_progress_weighted"].append(weighted_obj_prog_val)
        self.metrics["unified_progress"].append(self.max_unified_progress_ratio)
        self.metrics["deviation"].append(ee_min_dist)
        self.metrics["space_eff_cum"].append(s_eff_cum)
        self.metrics["time_eff_cum"].append(t_eff_cum)

        all_forces = np.linalg.norm(self.env.contact_forces(), axis=1)
        self.metrics["force"].append(np.max(all_forces) if len(all_forces)
                                     else 0.0)

        if info.get("success", False):
            self.metrics["success"] = True
        self.last_ee_pos = ee_pos
        _ = delta_prog  # kept for diagnostic parity

    def calculate_score(self) -> dict:
        if not self.metrics["unified_progress"]:
            return self._empty_score()

        final_prog = (1.0 if self.metrics["success"]
                       else self.metrics["unified_progress"][-1])
        score_comp = final_prog * 100.0

        raw_space = float(self.metrics["space_eff_cum"][-1])
        raw_time = float(self.metrics["time_eff_cum"][-1])
        eta_space = min(self.gamma_expert, max(0.0, raw_space))
        eta_time = min(self.gamma_expert, max(0.0, raw_time))
        score_space = 100.0 * eta_space
        score_time = 100.0 * eta_time

        score_smooth = sparc_score(np.asarray(self.velocity_history),
                                   fs=1.0 / self.dt)

        saf = safety_score(
            np.asarray(self.metrics["force"]),
            ee_positions=np.asarray(self.ee_positions) if self.ee_positions
            else None,
            workspace=self.workspace,
            num_steps=len(self.metrics["force"]),
            f_sus_lim=self.f_sus_lim, f_peak_lim=self.f_peak_lim)
        score_safety = saf["safety"]

        avg_dev = float(np.mean(self.metrics["deviation"]))
        total = ((score_comp * 0.45) + (score_space * 0.15)
                 + (score_time * 0.10) + (score_smooth * 0.15)
                 + (score_safety * 0.15))
        if self.metrics["success"]:
            total = max(total, 60.0)

        dropped = (self.metrics["ee_progress"][-1] > 0.9
                   and self.metrics["obj_progress_weighted"][-1] < 0.5)

        return {
            "total_fwdbias": round(total, 2),
            "total_hybrid": round(total, 2),
            "completion": round(score_comp, 2),
            "space_eff": round(score_space, 2),
            "time_eff": round(score_time, 2),
            "smoothness": round(score_smooth, 2),
            "safety": round(score_safety, 2),
            "raw_dev": round(avg_dev, 3),
            "raw_space_val": round(eta_space, 3),
            "raw_time_val": round(eta_time, 3),
            "raw_fft_val": round(score_smooth, 3),
            "p_kin": saf["p_kin"], "p_sem": saf["p_sem"], "p_dyn": saf["p_dyn"],
            "dropped": bool(dropped),
        }

    @staticmethod
    def _empty_score() -> dict:
        return {
            "total_fwdbias": 0.0, "total_hybrid": 0.0,
            "completion": 0.0, "space_eff": 0.0, "time_eff": 0.0,
            "smoothness": 0.0, "safety": 0.0, "raw_dev": 0.0,
            "raw_space_val": 0.0, "raw_time_val": 0.0,
            "raw_fft_val": 0.0, "dropped": False,
        }

    def plot_diagnostic_metrics(self, save_path):
        if not _HAS_MPL:
            return
        times = self.metrics["time"]
        if not times:
            return
        scores = self.calculate_score()

        fig, axs = plt.subplots(3, 2, figsize=(16, 18))
        status = " [DROPPED!]" if scores.get("dropped") else ""
        num_objs = len(self.tracked_objects)
        fig.suptitle(
            f"Multi-Object Trajectory Eval ({num_objs} Objs) | "
            f"Total: {scores['total_fwdbias']}{status}",
            fontsize=20, fontweight="bold")

        axs[0, 0].plot(times, self.metrics["ee_progress"], "r--",
                       alpha=0.3, label="Reach (Arm)")
        axs[0, 0].plot(times, self.metrics["obj_progress_weighted"], "b--",
                       alpha=0.3, label="Objs (Weighted)")
        axs[0, 0].plot(times, self.metrics["unified_progress"], "g-",
                       lw=2.5, label="Unified")
        axs[0, 0].set_title(f"1. Sequence Completion: {scores['completion']}%")
        axs[0, 0].set_ylim(0, 1.05)
        axs[0, 0].legend(loc="upper left")
        axs[0, 0].grid(True)

        axs[0, 1].plot(times, self.metrics["deviation"], "purple", lw=2)
        axs[0, 1].set_title(f"2. Deviation: {scores['raw_dev']}m")
        axs[0, 1].grid(True)

        axs[1, 0].plot(times, self.metrics["space_eff_cum"],
                       color="teal", lw=2)
        axs[1, 0].set_title(f"3. Space Eff (Score: {scores['space_eff']})")
        axs[1, 0].grid(True)

        axs[1, 1].plot(times, self.metrics["time_eff_cum"],
                       color="navy", lw=2)
        axs[1, 1].set_title(f"4. Time Eff (Score: {scores['time_eff']})")
        axs[1, 1].grid(True)

        axs[2, 0].plot(times, self.velocity_history, color="darkorange",
                       lw=1.4)
        axs[2, 0].set_title(f"5. Smoothness (SPARC): {scores['smoothness']}")
        axs[2, 0].grid(True)

        peak_val = np.max(self.metrics["force"])
        axs[2, 1].plot(times, self.metrics["force"], "crimson", lw=1.5,
                       label="Force")
        axs[2, 1].axhline(self.f_sus_lim, color="black", linestyle="--",
                          label=f"Sustained ({self.f_sus_lim:.0f}N)")
        axs[2, 1].axhline(self.f_peak_lim, color="red", linestyle=":",
                          label=f"Peak ({self.f_peak_lim:.0f}N)")
        axs[2, 1].set_title(f"6. Safety: {scores['safety']} "
                            f"(Peak: {peak_val:.1f}N)")
        axs[2, 1].legend(loc="upper right")
        axs[2, 1].grid(True)
        axs[2, 1].set_ylim(0, max(70, peak_val * 1.1))

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(save_path)
        plt.close()
