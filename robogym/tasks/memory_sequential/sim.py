"""Memory-Dependent Sequential Reasoning sims."""

from __future__ import annotations

import numpy as np

from ...envs.physics_sim import _TABLE_Z, PhysicsReasoningSim
from ..base import ReasoningTask

_Z = _TABLE_Z + 0.02
_COLORS = ["red", "green", "blue"]
_OCCLUDE_AFTER = 10


class ColorHanoiSim(PhysicsReasoningSim):
    """Tower-of-Hanoi disassembly that preserves the initial colour order."""

    category = "memory"

    def _setup(self) -> None:
        r = self.rng
        order = list(r.permutation(_COLORS))
        self._order = order
        self._n = len(order)
        self._pegx = {"src": -0.12, "aux": 0.0, "dst": 0.12}
        self._pegy = 0.06
        self.obj_names = [f"ring_{c}" for c in order]
        self.obj_mass = {k: 0.05 for k in self.obj_names}
        self.obj_pos = {}
        for lvl, c in enumerate(order):
            self.obj_pos[f"ring_{c}"] = np.array(
                [self._pegx["src"], self._pegy, _Z + 0.02 * lvl])
        self.instance = {
            "color_order": order,
            "description": ("rebuild the rings on the blue stand in the "
                            "original colour order (memorize it first)"),
        }

    def _task_reset(self):
        self._stacks = {"src": [f"ring_{c}" for c in self._order],
                        "aux": [], "dst": []}

    def _graspable(self, name: str) -> bool:
        for st in self._stacks.values():
            if st and st[-1] == name:
                return True
        return False

    def _task_obs(self) -> dict:
        if self.step_count < _OCCLUDE_AFTER:
            enc = np.array([_COLORS.index(c) for c in self._order],
                           dtype=float)
        else:
            enc = np.full(self._n, -1.0)
        return {"initial_color_order": enc}

    def _task_step(self) -> None:
        if self.grasped is None:
            for k in self.obj_names:
                p = self.obj_pos[k]
                held_anywhere = any(k in s for s in self._stacks.values())
                if held_anywhere:
                    continue
                for pk, px in self._pegx.items():
                    if (abs(p[0] - px) < 0.05
                            and abs(p[1] - self._pegy) < 0.05):
                        lvl = len(self._stacks[pk])
                        self.obj_pos[k] = np.array(
                            [px, self._pegy, _Z + 0.02 * lvl])
                        self._stacks[pk].append(k)
                        break
        else:
            for st in self._stacks.values():
                if st and st[-1] == self.grasped:
                    st.pop()
                    break

    def _success(self) -> bool:
        dst = self._stacks["dst"]
        if len(dst) != self._n or self.grasped is not None:
            return False
        got = [r.split("_")[1] for r in dst]
        return got == self._order

    def oracle_waypoints(self):
        """src -> aux reverses, then aux -> dst reverses back."""
        wps = []

        def move(frm, to):
            sx, dx = self._pegx[frm], self._pegx[to]
            wps.extend([
                [sx, self._pegy, _Z + 0.10, -1.0],
                [sx, self._pegy, _Z + 0.01, 1.0],
                [sx, self._pegy, _Z + 0.14, 1.0],
                [dx, self._pegy, _Z + 0.14, 1.0],
                [dx, self._pegy, _Z + 0.02, 1.0],
                [dx, self._pegy, _Z + 0.10, -1.0],
            ])
        for _ in range(self._n):
            move("src", "aux")
        for _ in range(self._n):
            move("aux", "dst")
        return wps

    def naive_waypoints(self):
        """Move every ring src -> dst directly, which reverses the colour order."""
        wps = []
        sx, dx = self._pegx["src"], self._pegx["dst"]
        for _ in range(self._n):
            wps.extend([
                [sx, self._pegy, _Z + 0.10, -1.0],
                [sx, self._pegy, _Z + 0.01, 1.0],
                [sx, self._pegy, _Z + 0.14, 1.0],
                [dx, self._pegy, _Z + 0.14, 1.0],
                [dx, self._pegy, _Z + 0.02, 1.0],
                [dx, self._pegy, _Z + 0.10, -1.0],
            ])
        return wps

    def _task_state(self):
        idx = {n: i for i, n in enumerate(self.obj_names)}
        flat = []
        for pk in ("src", "aux", "dst"):
            st = self._stacks.get(pk, [])
            flat += [idx[x] for x in st] + [-1] * (self._n - len(st))
        return np.array(flat, dtype=float)

    def _set_task_state(self, a):
        if len(a) < 3 * self._n:
            self._task_reset()
            return
        self._stacks = {}
        for bi, pk in enumerate(("src", "aux", "dst")):
            seg = a[bi * self._n:(bi + 1) * self._n]
            self._stacks[pk] = [self.obj_names[int(round(v))]
                                for v in seg if v >= 0]


class ColorHanoiTask(ReasoningTask):
    category = "memory"
    family = "color_hanoi"
    sim_cls = ColorHanoiSim


class CountingSim(PhysicsReasoningSim):
    """Place exactly ``count_a`` blocks in area A and ``count_b`` in area B."""

    category = "memory"

    def _setup(self) -> None:
        r = self.rng
        a = int(r.integers(1, 3))
        b = int(r.integers(1, 3))
        self._a, self._b = a, b
        total = a + b
        self.obj_names = [f"blk_{i}" for i in range(total)]
        self.obj_mass = {k: 0.08 for k in self.obj_names}
        self.obj_pos = {f"blk_{i}": np.array([-0.13 + 0.045 * i, -0.10, _Z])
                        for i in range(total)}
        self._regA = np.array([0.10, 0.17, _Z])
        self._regB = np.array([0.10, -0.03, _Z])
        self.instance = {
            "count_a": a, "count_b": b, "total": total,
            "description": ("distribute the blocks so each area holds exactly "
                            "the number shown on its ground marker"),
        }

    def _task_obs(self) -> dict:
        vis = self.step_count < _OCCLUDE_AFTER
        return {"ground_numbers": np.array(
            [float(self._a), float(self._b)] if vis else [-1.0, -1.0])}

    def _inA(self, p) -> bool:
        return bool(np.linalg.norm(p[:2] - self._regA[:2]) < 0.10)

    def _inB(self, p) -> bool:
        return bool(np.linalg.norm(p[:2] - self._regB[:2]) < 0.10)

    def _success(self) -> bool:
        if self.grasped is not None:
            return False
        na = sum(1 for k in self.obj_names if self._inA(self.obj_pos[k]))
        nb = sum(1 for k in self.obj_names if self._inB(self.obj_pos[k]))
        placed = sum(1 for k in self.obj_names
                     if self._inA(self.obj_pos[k])
                     or self._inB(self.obj_pos[k]))
        return bool(na == self._a and nb == self._b
                    and placed == len(self.obj_names))

    def oracle_waypoints(self):
        wps, idx = [], 0

        def place(name, reg, k):
            s = self.obj_pos[name]
            t = reg + np.array([0.025 * k, 0.0, 0.0])
            return [
                [s[0], s[1], s[2] + 0.06, -1.0], [s[0], s[1], s[2], 1.0],
                [s[0], s[1], s[2] + 0.10, 1.0],
                [t[0], t[1], t[2] + 0.10, 1.0], [t[0], t[1], t[2], 1.0],
                [t[0], t[1], t[2], -1.0],
            ]
        for k in range(self._a):
            wps += place(self.obj_names[idx], self._regA, k); idx += 1
        for k in range(self._b):
            wps += place(self.obj_names[idx], self._regB, k); idx += 1
        return wps

    def naive_waypoints(self):
        """Dump every block into area A only -> over-fills A, leaves B empty."""
        wps = []
        for i, k in enumerate(self.obj_names):
            s = self.obj_pos[k]
            t = self._regA + np.array([0.022 * i, 0.0, 0.0])
            wps += [
                [s[0], s[1], s[2] + 0.06, -1.0], [s[0], s[1], s[2], 1.0],
                [s[0], s[1], s[2] + 0.10, 1.0],
                [t[0], t[1], t[2] + 0.10, 1.0], [t[0], t[1], t[2], 1.0],
                [t[0], t[1], t[2], -1.0],
            ]
        return wps

    def _task_state(self):
        return np.array([self._a, self._b], dtype=float)

    def _set_task_state(self, arr):
        if len(arr) >= 2:
            self._a, self._b = int(round(arr[0])), int(round(arr[1]))


class SequentialCountingTask(ReasoningTask):
    category = "memory"
    family = "sequential_counting"
    sim_cls = CountingSim
