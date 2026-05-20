"""Dependency-free synthetic simulation backend.

A lightweight backend implementing the :class:`SimBackend` protocol in pure
numpy. a small tabletop pick-and-place world whose kinematics, moved-object
tracking, contact-force signal and flattened-state save/restore are faithful
enough to drive the metric engine end-to-end without mujoco/robosuite/LIBERO.

It lets the full pipeline run and be tested without the heavy simulation stack
(``python -m robogym.demo``, ``pytest``). For real LIBERO physics, use
:class:`robogym.envs.libero_env`.
"""

from __future__ import annotations

import numpy as np

# ee, gripper bodies first, then task objects. Names chosen so the
# evaluator's keyword blacklist keeps exactly the task objects as "moved".
_FIXED_BODIES = ["world", "robot0_base", "robot0_link7", "gripper0_grip_site"]

class SyntheticPickPlaceSim:
    """A minimal 7-DoF-delta tabletop pick-and-place world.

    Action convention matches LIBERO: ``[dx, dy, dz, dR, dP, dY, grip]`` with
    ``grip < 0`` == open, ``grip > 0`` == close (OpenVLA sign convention is
    handled upstream by the policy adapter).
    """

    def __init__(self, task_objects: list[str] | None = None,
                 target_name: str = "basket", seed: int = 0,
                 dt: float = 0.05, action_scale: float = 0.05):
        self.rng = np.random.default_rng(seed)
        self.dt = dt
        self.action_scale = action_scale
        self.task_objects = task_objects or ["alphabet_soup"]
        self.target_name = target_name

        self.body_list = list(_FIXED_BODIES) + list(self.task_objects) \
            + [target_name]
        self._ee_body_id = self.body_list.index("gripper0_grip_site")

        # nominal geometry (metres, robosuite-ish tabletop frame)
        self._home_ee = np.array([-0.10, 0.0, 1.05])
        self._obj_home = {
            name: np.array([0.0 + 0.04 * i, -0.10 + 0.05 * i, 0.92])
            for i, name in enumerate(self.task_objects)
        }
        self._target_pos = np.array([0.10, 0.20, 0.93])

        self._init_dynamic_state()

    # state
    def _init_dynamic_state(self):
        self.ee = self._home_ee.copy()
        self.gripper = -1.0
        self.obj_pos = {k: v.copy() for k, v in self._obj_home.items()}
        self.grasped = None
        self.step_count = 0
        self.success = False
        self._last_force = 0.0

    def reset(self) -> dict:
        self._init_dynamic_state()
        return self._obs()

    def _obs(self) -> dict:
        return {
            "robot0_eef_pos": self.ee.copy(),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.array([self.gripper, -self.gripper]),
            "robot0_joint_pos": np.zeros(7),
            **{f"{k}_pos": v.copy() for k, v in self.obj_pos.items()},
        }

    # SimBackend protocol
    @property
    def n_bodies(self) -> int:
        return len(self.body_list)

    def body_names(self) -> list[str]:
        return list(self.body_list)

    def ee_pos(self) -> np.ndarray:
        return self.ee.copy()

    def ee_velocity(self) -> np.ndarray:  # finite-diff fallback used otherwise
        return getattr(self, "_ee_vel", np.zeros(3))

    def body_xpos(self, body_id: int) -> np.ndarray:
        name = self.body_list[body_id]
        if name in self.obj_pos:
            return self.obj_pos[name].copy()
        if name == self.target_name:
            return self._target_pos.copy()
        if name == "gripper0_grip_site":
            return self.ee.copy()
        if name == "robot0_link7":
            return self.ee + np.array([0, 0, 0.08])
        if name == "robot0_base":
            return np.array([-0.55, 0.0, 0.90])
        return np.zeros(3)

    def get_state(self) -> np.ndarray:
        parts = [self.ee, [self.gripper]]
        for k in self.task_objects:
            parts.append(self.obj_pos[k])
        grasped_id = (-1 if self.grasped is None
                      else self.task_objects.index(self.grasped))
        parts.append([grasped_id, self.step_count,
                      1.0 if self.success else 0.0])
        return np.concatenate([np.asarray(p, float).ravel() for p in parts])

    def set_state(self, flat_state: np.ndarray) -> None:
        f = np.asarray(flat_state, dtype=float).ravel()
        i = 0
        self.ee = f[i:i + 3].copy(); i += 3
        self.gripper = float(f[i]); i += 1
        for k in self.task_objects:
            self.obj_pos[k] = f[i:i + 3].copy(); i += 3
        gid = int(round(f[i])); i += 1
        self.grasped = None if gid < 0 else self.task_objects[gid]
        self.step_count = int(round(f[i])); i += 1
        self.success = bool(round(f[i]))

    def forward(self) -> None:  # no derived quantities to recompute
        return None

    def contact_forces(self) -> np.ndarray:
        """``cfrc_ext[1:]`` analogue: small baseline noise + table/grasp spikes."""
        n = self.n_bodies - 1
        base = self.rng.normal(0.0, 1.5, size=(n, 6))
        # table-press spike if EE pushes below table height
        press = max(0.0, 0.92 - self.ee[2]) * 600.0
        # grasp transient
        grasp_spike = 18.0 if (self.grasped is not None
                               and self.step_count % 17 == 0) else 0.0
        base[self._ee_body_id - 1, :3] += press + grasp_spike
        self._last_force = press + grasp_spike
        return base

    # dynamics
    def step(self, action) -> tuple[dict, float, bool, dict]:
        a = np.asarray(action, dtype=float).ravel()
        prev_ee = self.ee.copy()
        self.ee = self.ee + a[:3] * self.action_scale
        # keep inside a generous box (no hard clip -> metrics see the violation)
        self.gripper = float(np.clip(a[6] if len(a) > 6 else self.gripper,
                                     -1.0, 1.0))
        self._ee_vel = (self.ee - prev_ee) / self.dt

        # grasp logic: closing near an object grabs it
        if self.grasped is None and self.gripper > 0.0:
            for name, p in self.obj_pos.items():
                if np.linalg.norm(self.ee - p) < 0.05:
                    self.grasped = name
                    break
        # release
        if self.grasped is not None and self.gripper <= 0.0:
            self.grasped = None

        # grasped object follows the gripper
        if self.grasped is not None:
            self.obj_pos[self.grasped] = self.ee + np.array([0.0, 0.0, -0.02])

        # success: an object released into the target zone
        done = False
        for name, p in self.obj_pos.items():
            if (self.grasped != name
                    and np.linalg.norm(p[:2] - self._target_pos[:2]) < 0.06
                    and abs(p[2] - self._target_pos[2]) < 0.05):
                self.success = True
                done = True
        self.step_count += 1
        info = {"success": self.success}
        return self._obs(), (1.0 if self.success else 0.0), done, info

    # expert demo generator (for oracle policy + evaluator pre-roll)
    def generate_expert_demo(self, jitter: float = 0.0,
                             seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(expert_actions, expert_init_state)`` for a clean reach ->
        grasp -> transport -> place trajectory of the first task object.

        ``jitter`` injects small per-step noise (used to synthesise the 500
        balanced demos per task the paper mentions in §4.1)."""
        rng = np.random.default_rng(self.rng.integers(1 << 30)
                                    if seed is None else seed)
        self.reset()
        init_state = self.get_state()

        obj = self.task_objects[0]
        obj_p = self._obj_home[obj].copy()
        actions = []

        def move_towards(target, n, grip, jit):
            """Cosine-eased segment -> continuous, jerk-free EE velocity (paper
            §4.1: oracle waypoints "B-spline interpolated to produce
            continuous, jerk-free end-effector velocities"). Smooth velocity
            ramps keep the FFT/SPARC smoothness metric near ceiling
            for a true expert."""
            target = np.asarray(target, float)
            p0 = self.ee.copy()
            for k in range(1, n + 1):
                s = 0.5 - 0.5 * np.cos(np.pi * k / n)        # smoothstep ease
                desired = p0 + (target - p0) * s
                delta = (desired - self.ee) / self.action_scale
                if jit:
                    delta = delta + rng.normal(0, jit, 3)
                delta = np.clip(delta, -1.0, 1.0)
                act = np.array([delta[0], delta[1], delta[2], 0, 0, 0, grip])
                actions.append(act)
                self.step(act.tolist())

        move_towards(obj_p + np.array([0, 0, 0.06]), 26, -1.0, jitter)  # approach
        move_towards(obj_p, 12, -1.0, jitter)                           # descend
        move_towards(obj_p, 4, 1.0, 0.0)                                # grasp
        move_towards(obj_p + np.array([0, 0, 0.12]), 14, 1.0, jitter)   # lift
        move_towards(self._target_pos + np.array([0, 0, 0.10]), 30,
                     1.0, jitter)                                       # carry
        move_towards(self._target_pos, 12, 1.0, jitter)                 # lower
        move_towards(self._target_pos, 6, -1.0, 0.0)                    # release

        expert_actions = np.array(actions, dtype=float)
        self.reset()
        return expert_actions, init_state
