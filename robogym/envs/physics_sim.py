"""Physics-grounded reasoning sim (numpy only).

Subclasses declare objects via :meth:`_setup` and supply per-task hooks
(:meth:`_task_step`, :meth:`_success`, :meth:`oracle_waypoints`). Task
success is gated on physical / logical state so it cannot be solved by
replaying a memorised action sequence.
"""

from __future__ import annotations

import numpy as np

from ._smoothing import bspline_sample, grip_schedule

_FIXED = ["world", "robot0_base", "robot0_link7", "gripper0_grip_site"]
_EE_HOME = np.array([-0.10, 0.0, 1.05])
_TABLE_Z = 0.90
_G = 9.81


class PhysicsReasoningSim:
    """Rigid-body tabletop world with grasp dynamics and task hooks."""

    def __init__(self, seed: int = 0, dt: float = 0.05,
                 action_scale: float = 0.05):
        self.rng = np.random.default_rng(seed)
        self.dt = dt
        self.action_scale = action_scale

        self.obj_names: list[str] = []
        self.obj_pos: dict[str, np.ndarray] = {}
        self.obj_mass: dict[str, float] = {}
        self.obj_friction: dict[str, float] = {}
        self.obj_meta: dict[str, dict] = {}
        self.target_name = "goal"
        self._target_pos = np.array([0.12, 0.20, _TABLE_Z + 0.02])

        self.instance: dict = {}
        self._setup()

        self.body_list = list(_FIXED) + list(self.obj_names) + [self.target_name]
        self._ee_id = self.body_list.index("gripper0_grip_site")
        self._home_pos = {k: v.copy() for k, v in self.obj_pos.items()}
        self._init_dynamic()

    # Subclass hooks
    def _setup(self) -> None:
        raise NotImplementedError

    def _task_step(self) -> None:
        pass

    def _success(self) -> bool:
        return False

    def oracle_waypoints(self) -> list[np.ndarray]:
        raise NotImplementedError

    # State
    def _init_dynamic(self):
        self.ee = _EE_HOME.copy()
        self.gripper = -1.0
        self.grasped: str | None = None
        self.obj_pos = {k: v.copy() for k, v in self._home_pos.items()}
        self.step_count = 0
        self.success = False
        self._ee_vel = np.zeros(3)
        self._last_force = 0.0
        self._task_reset()

    def _task_reset(self) -> None:
        pass

    def reset(self) -> dict:
        self._init_dynamic()
        return self._obs()

    def _obs(self) -> dict:
        o = {
            "robot0_eef_pos": self.ee.copy(),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.array([self.gripper, -self.gripper]),
            "robot0_joint_pos": np.zeros(7),
        }
        for k, v in self.obj_pos.items():
            o[f"{k}_pos"] = v.copy()
        o.update(self._task_obs())
        return o

    def _task_obs(self) -> dict:
        return {}

    # SimBackend protocol
    @property
    def n_bodies(self) -> int:
        return len(self.body_list)

    def body_names(self) -> list[str]:
        return list(self.body_list)

    def ee_pos(self) -> np.ndarray:
        return self.ee.copy()

    def ee_velocity(self) -> np.ndarray:
        return self._ee_vel.copy()

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
            return np.array([-0.55, 0.0, _TABLE_Z])
        return np.zeros(3)

    def get_state(self) -> np.ndarray:
        parts = [self.ee, [self.gripper]]
        for k in self.obj_names:
            parts.append(self.obj_pos[k])
        gid = -1 if self.grasped is None else self.obj_names.index(self.grasped)
        parts.append([gid, self.step_count, 1.0 if self.success else 0.0])
        parts.append(self._task_state())
        return np.concatenate([np.asarray(p, float).ravel() for p in parts])

    def set_state(self, flat: np.ndarray) -> None:
        f = np.asarray(flat, float).ravel()
        i = 0
        self.ee = f[i:i + 3].copy(); i += 3
        self.gripper = float(f[i]); i += 1
        for k in self.obj_names:
            self.obj_pos[k] = f[i:i + 3].copy(); i += 3
        gid = int(round(f[i])); i += 1
        self.grasped = None if gid < 0 else self.obj_names[gid]
        self.step_count = int(round(f[i])); i += 1
        self.success = bool(round(f[i])); i += 1
        self._set_task_state(f[i:])

    def _task_state(self) -> np.ndarray:
        return np.zeros(0)

    def _set_task_state(self, arr: np.ndarray) -> None:
        return None

    def forward(self) -> None:
        return None

    def contact_forces(self) -> np.ndarray:
        n = self.n_bodies - 1
        base = self.rng.normal(0.0, 1.2, size=(n, 6))
        press = max(0.0, _TABLE_Z - self.ee[2]) * 650.0
        spike = 16.0 if (self.grasped is not None
                         and self.step_count % 19 == 0) else 0.0
        base[self._ee_id - 1, :3] += press + spike
        self._last_force = press + spike
        return base

    # Dynamics
    def step(self, action):
        a = np.asarray(action, float).ravel()
        prev = self.ee.copy()
        self.ee = self.ee + a[:3] * self.action_scale
        self.gripper = float(np.clip(a[6] if len(a) > 6 else self.gripper,
                                     -1.0, 1.0))
        self._ee_vel = (self.ee - prev) / self.dt

        if self.grasped is None and self.gripper > 0.0:
            near = [(np.linalg.norm(self.ee - p), k)
                    for k, p in self.obj_pos.items()
                    if self._graspable(k)]
            if near:
                d, k = min(near)
                if d < 0.05:
                    self.grasped = k
        if self.grasped is not None and self.gripper <= 0.0:
            self.grasped = None

        if self.grasped is not None:
            self.obj_pos[self.grasped] = self.ee + np.array([0, 0, -0.02])

        for k, p in self.obj_pos.items():
            if k == self.grasped:
                continue
            floor = self._support_height(k)
            if p[2] > floor + 1e-4:
                p[2] = max(floor, p[2] - _G * self.dt * self.dt * 6.0)
            else:
                p[2] = floor

        self._task_step()
        if not self.success and self._success():
            self.success = True
        self.step_count += 1
        info = {"success": self.success}
        return self._obs(), (1.0 if self.success else 0.0), self.success, info

    def _graspable(self, name: str) -> bool:
        return True

    def _support_height(self, name: str) -> float:
        return _TABLE_Z + 0.02

    # Expert demo
    def generate_expert_demo(self, jitter: float = 0.0,
                             seed: int | None = None,
                             samples_per_segment: int = 8
                             ) -> tuple[np.ndarray, np.ndarray]:
        """Sample a clamped cubic B-spline through the oracle waypoints
        and execute the resulting actions on this sim (Sec. 4.1)."""
        rng = np.random.default_rng(self.rng.integers(1 << 30)
                                    if seed is None else seed)
        self.reset()
        init_state = self.get_state()
        wps = self.oracle_waypoints()
        if not wps:
            self.reset()
            return np.empty((0, 7), float), init_state

        carrying = False
        grips_at_wp = [-1.0]
        pts = [self.ee.copy()]
        for wp in wps:
            pts.append(np.asarray(wp[:3], float))
            grip = float(wp[3]) if len(wp) > 3 else (1.0 if carrying else -1.0)
            grips_at_wp.append(grip)
            carrying = grip > 0
        pts = np.asarray(pts, float)
        grips_at_wp = np.asarray(grips_at_wp, float)

        positions = bspline_sample(pts, samples_per_segment=samples_per_segment)
        grips = grip_schedule(len(pts), samples_per_segment, grips_at_wp)

        actions = []
        for target_pos, grip in zip(positions, grips):
            delta = (target_pos - self.ee) / self.action_scale
            if jitter:
                delta = delta + rng.normal(0.0, jitter, 3)
            delta = np.clip(delta, -1.0, 1.0)
            g = grip
            if grip > 0 and self.grasped is None:
                near = [(np.linalg.norm(self.ee - p), kk)
                        for kk, p in self.obj_pos.items()
                        if self._graspable(kk)]
                if near and min(near)[0] < 0.05:
                    g = 1.0
            act = np.array([delta[0], delta[1], delta[2], 0, 0, 0, g])
            actions.append(act)
            self.step(act.tolist())

        self.reset()
        return np.asarray(actions, float), init_state

    def generate_naive_demo(self, seed: int | None = None,
                            samples_per_segment: int = 8
                            ) -> tuple[np.ndarray, np.ndarray]:
        """Non-reasoning baseline: same B-spline machinery but with
        :meth:`naive_waypoints` so the demo provably fails the success
        predicate when reasoning is required."""
        if not hasattr(self, "naive_waypoints"):
            raise NotImplementedError
        self.reset()
        init_state = self.get_state()
        wps = self.naive_waypoints()
        if not wps:
            self.reset()
            return np.empty((0, 7), float), init_state

        carrying = False
        grips_at_wp = [-1.0]
        pts = [self.ee.copy()]
        for wp in wps:
            pts.append(np.asarray(wp[:3], float))
            grip = float(wp[3]) if len(wp) > 3 else (1.0 if carrying else -1.0)
            grips_at_wp.append(grip)
            carrying = grip > 0
        pts = np.asarray(pts, float)
        grips_at_wp = np.asarray(grips_at_wp, float)

        positions = bspline_sample(pts, samples_per_segment=samples_per_segment)
        grips = grip_schedule(len(pts), samples_per_segment, grips_at_wp)

        actions = []
        for target_pos, grip in zip(positions, grips):
            delta = (target_pos - self.ee) / self.action_scale
            delta = np.clip(delta, -1.0, 1.0)
            g = grip
            if grip > 0 and self.grasped is None:
                near = [(np.linalg.norm(self.ee - p), kk)
                        for kk, p in self.obj_pos.items()
                        if self._graspable(kk)]
                if near and min(near)[0] < 0.05:
                    g = 1.0
            act = np.array([delta[0], delta[1], delta[2], 0, 0, 0, g])
            actions.append(act)
            self.step(act.tolist())

        self.reset()
        return np.asarray(actions, float), init_state
