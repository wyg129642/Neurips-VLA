"""Physical Intuition Reasoning sims (paper §3.6, Fig 2).

SeesawSim: a lever-torque balance with two blocks of hidden, test-time
randomized masses; the visible seesaw tilt equals the sign of the net torque.
The agent must deduce the heavier block from the tilt and place it on the
matching plate. A policy that ignores the tilt is correct 50% of the time.

MazeSim: the maze geometry from ``maze.xml`` (walls including the central
``w_mid``) with real wall collision. A straight-line carry hits ``w_mid`` and
fails; only a start->mid->target detour succeeds. Progress is the two-segment
arc length along that detour.
"""

from __future__ import annotations

import numpy as np

from ...envs.physics_sim import _TABLE_Z, PhysicsReasoningSim
from ..base import ReasoningTask

_Z = _TABLE_Z + 0.02

def _polyline_progress(pos, pts) -> float:
    """Normalised arc-length of the closest point on polyline ``pts`` to
    ``pos`` (xy). Independent of axis flips."""
    pts = [np.asarray(p, float)[:2] for p in pts]
    p = np.asarray(pos, float)[:2]
    seg_len = [np.linalg.norm(pts[i + 1] - pts[i]) + 1e-9
               for i in range(len(pts) - 1)]
    total = sum(seg_len)
    best_d, best_arc = np.inf, 0.0
    acc = 0.0
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        ab = b - a
        t = float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0))
        proj = a + t * ab
        d = np.linalg.norm(p - proj)
        if d < best_d:
            best_d = d
            best_arc = acc + t * seg_len[i]
        acc += seg_len[i]
    return float(np.clip(best_arc / total, 0.0, 1.0))

# Seesaw weight comparison
class SeesawSim(PhysicsReasoningSim):
    category = "physical"

    def _setup(self) -> None:
        r = self.rng
        m_a, m_b = float(r.uniform(0.15, 1.0)), float(r.uniform(0.15, 1.0))
        while abs(m_a - m_b) < 0.12:                 # keep it decidable
            m_b = float(r.uniform(0.15, 1.0))
        self.obj_names = ["block_a", "block_b"]
        self.obj_pos = {
            "block_a": np.array([-0.06, -0.12, _Z]),
            "block_b": np.array([0.06, -0.12, _Z]),
        }
        self.obj_mass = {"block_a": m_a, "block_b": m_b}
        self.obj_friction = {"block_a": 1.0, "block_b": 1.0}
        heavier = "block_a" if m_a > m_b else "block_b"
        # heavy plate on the left, light plate on the right (randomize side)
        side = float(r.choice([-1.0, 1.0]))
        self._heavy_target = np.array([0.13 * side, 0.20, _Z])
        self._light_target = np.array([-0.13 * side, 0.20, _Z])
        self._heavier = heavier
        self.instance = {
            "mass_a": m_a, "mass_b": m_b, "heavier": heavier,
            "heavy_target": self._heavy_target, "side": side,
            "description": ("read the seesaw tilt to find the heavier block "
                            "and place it on the reinforced plate"),
        }

    def _task_obs(self) -> dict:
        # visible tilt: positive => block_a side lower => block_a heavier
        tilt = np.tanh(2.5 * (self.obj_mass["block_a"]
                              - self.obj_mass["block_b"]))
        return {"seesaw_tilt": np.array([tilt], dtype=float)}

    def _success(self) -> bool:
        hp = self.obj_pos[self._heavier]
        light = "block_b" if self._heavier == "block_a" else "block_a"
        lp = self.obj_pos[light]
        on_heavy = (np.linalg.norm(hp[:2] - self._heavy_target[:2]) < 0.06
                    and self.grasped != self._heavier)
        light_not_there = np.linalg.norm(
            lp[:2] - self._heavy_target[:2]) > 0.08
        return bool(on_heavy and light_not_there)

    def oracle_waypoints(self):
        h = self.obj_pos[self._heavier]
        return [
            [h[0], h[1], h[2] + 0.06, -1.0],            # approach (open)
            [h[0], h[1], h[2], 1.0],                    # grasp
            [h[0], h[1], h[2] + 0.12, 1.0],             # lift
            [self._heavy_target[0], self._heavy_target[1],
             self._heavy_target[2] + 0.10, 1.0],        # carry
            [self._heavy_target[0], self._heavy_target[1],
             self._heavy_target[2], 1.0],               # lower
            [self._heavy_target[0], self._heavy_target[1],
             self._heavy_target[2], -1.0],              # release
        ]

    def naive_waypoints(self):
        """Non-reasoning baseline: always grab block_a, ignore the tilt cue
        (correct only ~50 % of the time -> proves reasoning is required)."""
        a = self.obj_pos["block_a"]
        return [
            [a[0], a[1], a[2] + 0.06, -1.0], [a[0], a[1], a[2], 1.0],
            [a[0], a[1], a[2] + 0.12, 1.0],
            [self._heavy_target[0], self._heavy_target[1],
             self._heavy_target[2] + 0.10, 1.0],
            [self._heavy_target[0], self._heavy_target[1],
             self._heavy_target[2], 1.0],
            [self._heavy_target[0], self._heavy_target[1],
             self._heavy_target[2], -1.0],
        ]

    def _task_state(self):
        return np.array([self.obj_mass["block_a"], self.obj_mass["block_b"]])

    def _set_task_state(self, a):
        if len(a) >= 2:
            self.obj_mass["block_a"] = float(a[0])
            self.obj_mass["block_b"] = float(a[1])

class SeesawWeightTask(ReasoningTask):
    category = "physical"
    family = "seesaw_weight"
    sim_cls = SeesawSim

# Maze navigation: geometry intent plus wall collision.
# A central barrier spans the LEFT side at y=0 and leaves a gap on the
# RIGHT. A straight-line carry crosses the barrier and fails; the only
# solution is to detour through the right gap.
_OUTER = 0.15
_GAP_X = 0.05               # barrier spans x in [-_OUTER, _GAP_X]; gap to right
_BARRIER_Y = 0.0
_BARRIER_HY = 0.012

class MazeSim(PhysicsReasoningSim):
    category = "physical"

    def _setup(self) -> None:
        r = self.rng
        mirror = float(r.choice([-1.0, 1.0]))        # flip start/target side
        self._mirror = mirror
        self._start = np.array([-0.10 * mirror, 0.11, _Z])
        self._tgt = np.array([-0.10 * mirror, -0.11, _Z])
        self._gap = np.array([0.10, 0.0, _Z])        # the right-side opening
        # detour via-points: start -> top-right -> through gap -> bottom -> target
        self._mid = np.array([0.10, 0.11, _Z])
        self.obj_names = ["ball"]
        self.obj_pos = {"ball": self._start.copy()}
        self.obj_mass = {"ball": 0.05}
        self.obj_friction = {"ball": 0.6}
        self._target_pos = self._tgt.copy()
        self.target_name = "goal"
        self.instance = {
            "mirror": mirror, "start": self._start, "mid": self._mid,
            "gap": self._gap, "goal": self._tgt,
            "description": ("guide the ball to the target; a wall blocks the "
                            "direct path so detour through the side opening"),
        }

    @staticmethod
    def _crosses_barrier(old, new) -> bool:
        """True if segment old->new crosses the y=0 barrier within its
        x-extent (x <= _GAP_X). Outside that (the right gap) it's free."""
        if (old[1] - _BARRIER_Y) * (new[1] - _BARRIER_Y) > 0:
            return False                              # no y=0 crossing
        denom = (new[1] - old[1])
        if abs(denom) < 1e-9:
            return abs(old[1] - _BARRIER_Y) <= _BARRIER_HY and old[0] <= _GAP_X
        t = (_BARRIER_Y - old[1]) / denom
        x_cross = old[0] + t * (new[0] - old[0])
        return x_cross <= _GAP_X                      # within barrier (no gap)

    def _task_step(self) -> None:
        b = self.obj_pos["ball"]
        if self.grasped == "ball":
            old = getattr(self, "_prev_ball", b.copy())
            if self._crosses_barrier(old, b):
                # collision: ball cannot pass; it stays, grasp is lost
                self.obj_pos["ball"] = old.copy()
                self.grasped = None
                b = self.obj_pos["ball"]
        b[0] = float(np.clip(b[0], -_OUTER, _OUTER))
        b[1] = float(np.clip(b[1], -_OUTER, _OUTER))
        self._prev_ball = self.obj_pos["ball"].copy()

    def _task_reset(self) -> None:
        self._prev_ball = self.obj_pos["ball"].copy()

    def _success(self) -> bool:
        b = self.obj_pos["ball"]
        return bool(np.linalg.norm(b[:2] - self._tgt[:2]) < 0.05
                    and self.grasped != "ball")

    def progress(self) -> float:
        """Fraction of the detour polyline (start->mid->gap->target) traversed."""
        return _polyline_progress(
            self.obj_pos["ball"],
            [self._start, self._mid, self._gap, self._tgt])

    def oracle_waypoints(self):
        s, m, g, t = self._start, self._mid, self._gap, self._tgt
        return [
            [s[0], s[1], s[2] + 0.05, -1.0],
            [s[0], s[1], s[2], 1.0],                    # grab ball
            [m[0], m[1], m[2], 1.0],                    # to top-right
            [g[0], g[1], g[2], 1.0],                    # through the gap
            [t[0], t[1], t[2], 1.0],                    # down to target
            [t[0], t[1], t[2], -1.0],                   # release
        ]

    def naive_waypoints(self):
        """Non-reasoning baseline: straight line ball->target. Crosses the
        central ``w_mid`` wall -> blocked -> fails."""
        s, t = self._start, self._tgt
        return [
            [s[0], s[1], s[2] + 0.05, -1.0], [s[0], s[1], s[2], 1.0],
            [t[0], t[1], t[2], 1.0], [t[0], t[1], t[2], -1.0],
        ]

    def _task_state(self):
        return np.array([self._mirror])

    def _set_task_state(self, a):
        if len(a):
            self._mirror = float(a[0])

class MazeTask(ReasoningTask):
    category = "physical"
    family = "maze_navigation"
    sim_cls = MazeSim

    @staticmethod
    def get_progress(local_pos, instance):
        """maze progress: fraction of the detour polyline
        (start->mid->gap->target) traversed; 0 if the ball fell off the table."""
        if local_pos[2] < _TABLE_Z - 0.05:            # fell off the table
            return 0.0
        gap = instance.get("gap", instance["mid"])
        return _polyline_progress(local_pos, [instance["start"],
                                              instance["mid"], gap,
                                              instance["goal"]])
