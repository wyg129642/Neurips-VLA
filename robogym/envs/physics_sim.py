"""Physics-grounded reasoning simulator (dependency-free).

The System-2 tasks (§3.6) require physical/logical reasoning: success
must follow from state (torque balance, stacking legality, containment
count, geometric fit, wall collision), not from replaying a scripted
answer, and physical properties (mass, friction) are randomised at test
time and affect outcomes.

This module provides a small rigid-body world with gravity, support,
grasp/carry, per-task dynamics hooks, and logical success predicates. It
is deliberately not MuJoCo (which is an optional heavy dependency).
The matching MuJoCo / robosuite environments live in
``robogym/tasks/**/sim_robosuite.py``.
"""

from __future__ import annotations

import numpy as np

_FIXED = ["world", "robot0_base", "robot0_link7", "gripper0_grip_site"]
_EE_HOME = np.array([-0.10, 0.0, 1.05])
_TABLE_Z = 0.90
_G = 9.81

class PhysicsReasoningSim:
    """Rigid-body tabletop world with gravity, support, grasp and task hooks.

    Subclasses declare objects via :meth:`_setup` and implement
    :meth:`_task_step` (extra per-step dynamics) and :meth:`_success`
    (logical/physical goal predicate). Implements the full
    :class:`robogym.envs.base.SimBackend` protocol so the trajectory
    evaluator scores these unchanged.
    """

    def __init__(self, seed: int = 0, dt: float = 0.05,
                 action_scale: float = 0.05):
        self.rng = np.random.default_rng(seed)
        self.dt = dt
        self.action_scale = action_scale

        # object registry
        self.obj_names: list[str] = []
        self.obj_pos: dict[str, np.ndarray] = {}
        self.obj_mass: dict[str, float] = {}
        self.obj_friction: dict[str, float] = {}
        self.obj_meta: dict[str, dict] = {}
        self.target_name = "goal"
        self._target_pos = np.array([0.12, 0.20, _TABLE_Z + 0.02])

        self.instance: dict = {}
        self._setup()  # subclass populates objects + self.instance

        self.body_list = list(_FIXED) + list(self.obj_names) + [
            self.target_name]
        self._ee_id = self.body_list.index("gripper0_grip_site")
        self._home_pos = {k: v.copy() for k, v in self.obj_pos.items()}
        self._init_dynamic()

    # subclass hooks
    def _setup(self) -> None:
        """Declare objects (name, pos, mass, friction, meta) and set
        :attr:`instance`.

        Test-time randomization (§3.6) belongs here: resample masses,
        positions, and orderings so memorisation fails and physics
        matters.
        """
        raise NotImplementedError

    def _task_step(self) -> None:
        """Extra per-step task dynamics (seesaw tilt, peg snapping, ...)."""

    def _success(self) -> bool:
        """Logical/physical goal predicate (the genuine reasoning check)."""
        return False

    def oracle_waypoints(self) -> list[np.ndarray]:
        """Algorithmic-oracle sub-goal cartesian waypoints solving this
        randomised instance (paper §4.1). The policy is not given these."""
        raise NotImplementedError

    # core state
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
        """Optional subclass per-episode runtime reset (e.g. peg stacks)."""

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
        """Subclass exposes the reasoning cue here (seesaw angle, ground
        number, initial colour order, ...) so a reasoning policy can read
        it from the observation while a non-reasoning policy cannot."""
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
        gid = -1 if self.grasped is None else self.obj_names.index(
            self.grasped)
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

    # dynamics: integrate EE, grasp, gravity+support, task, success
    def step(self, action):
        a = np.asarray(action, float).ravel()
        prev = self.ee.copy()
        self.ee = self.ee + a[:3] * self.action_scale
        self.gripper = float(np.clip(a[6] if len(a) > 6 else self.gripper,
                                     -1.0, 1.0))
        self._ee_vel = (self.ee - prev) / self.dt

        # grasp nearest object when closing; release when opening
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

        # gravity + flat-table support for free objects
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

    # overridable physics helpers
    def _graspable(self, name: str) -> bool:
        return True

    def _support_height(self, name: str) -> float:
        """Z the object rests at when unsupported (default: table top)."""
        return _TABLE_Z + 0.02

    # expert demo: oracle waypoints -> cosine-eased jerk-free actions
    def generate_expert_demo(self, jitter: float = 0.0,
                             seed: int | None = None
                             ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.rng.integers(1 << 30)
                                    if seed is None else seed)
        self.reset()
        init_state = self.get_state()
        actions: list[np.ndarray] = []

        def seg(target, n, grip, jit):
            target = np.asarray(target, float)
            p0 = self.ee.copy()
            for k in range(1, n + 1):
                s = 0.5 - 0.5 * np.cos(np.pi * k / n)
                desired = p0 + (target - p0) * s
                d = (desired - self.ee) / self.action_scale
                if jit:
                    d = d + rng.normal(0, jit, 3)
                d = np.clip(d, -1, 1)
                g = grip
                # auto-grasp when first reaching a graspable object
                if grip > 0 and self.grasped is None:
                    near = [(np.linalg.norm(self.ee - p), kk)
                            for kk, p in self.obj_pos.items()
                            if self._graspable(kk)]
                    if near and min(near)[0] < 0.05:
                        g = 1.0
                act = np.array([d[0], d[1], d[2], 0, 0, 0, g])
                actions.append(act)
                self.step(act.tolist())

        self._roll_waypoints(self.oracle_waypoints(), seg)
        self.reset()
        return np.asarray(actions, float), init_state

    @staticmethod
    def _roll_waypoints(wps, seg) -> None:
        carrying = False
        for wp in wps:
            grip = wp[3] if len(wp) > 3 else (1.0 if carrying else -1.0)
            seg(np.asarray(wp[:3], float), 8, grip, 0.0)
            carrying = grip > 0

    def generate_naive_demo(self, seed: int | None = None
                            ) -> tuple[np.ndarray, np.ndarray]:
        """Non-reasoning baseline trajectory (uses :meth:`naive_waypoints`).

        Used as a check that the task genuinely requires reasoning: the
        oracle should succeed while this baseline should fail, because
        success is a physical/logical predicate rather than script replay.
        """
        if not hasattr(self, "naive_waypoints"):
            raise NotImplementedError
        self.reset()
        init_state = self.get_state()
        actions: list[np.ndarray] = []

        def seg(target, n, grip, jit):
            p0 = self.ee.copy()
            for k in range(1, n + 1):
                s = 0.5 - 0.5 * np.cos(np.pi * k / n)
                d = ((p0 + (np.asarray(target, float) - p0) * s) - self.ee) \
                    / self.action_scale
                g = grip
                if grip > 0 and self.grasped is None:
                    near = [(np.linalg.norm(self.ee - p), kk)
                            for kk, p in self.obj_pos.items()
                            if self._graspable(kk)]
                    if near and min(near)[0] < 0.05:
                        g = 1.0
                act = np.array([*np.clip(d, -1, 1), 0, 0, 0, g])
                actions.append(act)
                self.step(act.tolist())

        self._roll_waypoints(self.naive_waypoints(), seg)
        self.reset()
        return np.asarray(actions, float), init_state
