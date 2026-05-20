"""Symbolic and Geometric Reasoning sims.

TangramSim: N shaped pieces, N target slots each demanding a specific shape.
Success requires every piece to sit in the slot whose required shape matches
(geometric fit); placing pieces arbitrarily fails. The slot-to-shape assignment
is randomized per episode.

NumberBlockSim: a 3x3 Latin square with several empty cells; the agent holds
one block of a fixed value and must place it in the unique empty cell that
keeps every row and column repeat-free. The grid is randomized per episode.
"""

from __future__ import annotations

import numpy as np

from ...envs.physics_sim import _TABLE_Z, PhysicsReasoningSim
from ..base import ReasoningTask

_Z = _TABLE_Z + 0.02
_SHAPES = ["triangle", "square", "parallelogram"]

# Tangram assembly
class TangramSim(PhysicsReasoningSim):
    category = "geometric"

    def _setup(self) -> None:
        r = self.rng
        n = 3
        shapes = list(r.permutation(_SHAPES))[:n]
        # slot i demands shape slot_shape[i]; randomize the matching
        slot_shape = list(r.permutation(shapes))
        slots = np.array([[-0.08, 0.16, _Z], [0.0, 0.16, _Z],
                          [0.08, 0.16, _Z]])
        self.obj_names = [f"piece_{i}" for i in range(n)]
        self.obj_pos = {f"piece_{i}": np.array(
            [-0.10 + 0.10 * i, -0.12, _Z]) for i in range(n)}
        self.obj_mass = {k: 0.1 for k in self.obj_names}
        self._piece_shape = {f"piece_{i}": shapes[i] for i in range(n)}
        self._slots = slots
        self._slot_shape = slot_shape
        # correct slot index for each piece
        self._target = {}
        for i in range(n):
            sh = self._piece_shape[f"piece_{i}"]
            self._target[f"piece_{i}"] = slots[slot_shape.index(sh)]
        self.instance = {
            "piece_shape": self._piece_shape,
            "slot_shape": slot_shape,
            "description": ("place each tangram piece into the slot whose "
                            "shape matches to complete the figure"),
        }

    def _task_obs(self) -> dict:
        # the geometry cue: which shape each slot expects (one-hot per slot)
        enc = np.array([_SHAPES.index(s) for s in self._slot_shape],
                       dtype=float)
        return {"slot_shapes": enc}

    def _success(self) -> bool:
        for k in self.obj_names:
            if self.grasped == k:
                return False
            if np.linalg.norm(self.obj_pos[k][:2]
                              - self._target[k][:2]) > 0.05:
                return False
        return True

    def oracle_waypoints(self):
        wps = []
        for k in self.obj_names:
            s = self.obj_pos[k]
            t = self._target[k]
            wps += [
                [s[0], s[1], s[2] + 0.06, -1.0],
                [s[0], s[1], s[2], 1.0],
                [s[0], s[1], s[2] + 0.10, 1.0],
                [t[0], t[1], t[2] + 0.10, 1.0],
                [t[0], t[1], t[2], 1.0],
                [t[0], t[1], t[2], -1.0],
            ]
        return wps

    def naive_waypoints(self):
        """Non-reasoning baseline: drop every piece into slot 0 (ignores the
        shape<->slot constraint) -> fails."""
        wps, t = [], self._slots[0]
        for k in self.obj_names:
            s = self.obj_pos[k]
            wps += [
                [s[0], s[1], s[2] + 0.06, -1.0], [s[0], s[1], s[2], 1.0],
                [s[0], s[1], s[2] + 0.10, 1.0],
                [t[0], t[1], t[2] + 0.10, 1.0], [t[0], t[1], t[2], 1.0],
                [t[0], t[1], t[2], -1.0],
            ]
        return wps

class TangramTask(ReasoningTask):
    category = "geometric"
    family = "tangram_assembly"
    sim_cls = TangramSim

# Number-block Latin-square placement.
class NumberBlockSim(PhysicsReasoningSim):
    category = "geometric"

    def _setup(self) -> None:
        r = self.rng
        n = 3
        shift = int(r.integers(n))
        full = np.array([[(row + col + shift) % n + 1 for col in range(n)]
                         for row in range(n)])
        cells = [(i, j) for i in range(n) for j in range(n)]

        def legal_set(grid, blanks, v):
            """Blanks where placing value ``v`` keeps every row & column
            repeat-free given the currently filled cells."""
            out = []
            for (bi, bj) in blanks:
                row_ok = v not in [grid[bi, c] for c in range(n)
                                   if c != bj and grid[bi, c] != 0]
                col_ok = v not in [grid[rr, bj] for rr in range(n)
                                   if rr != bi and grid[rr, bj] != 0]
                if row_ok and col_ok:
                    out.append((bi, bj))
            return out

        # Search for an instance where the held value fits EXACTLY ONE blank
        # and that legal cell is NOT the first blank (so the non-reasoning
        # "drop in the first hole" baseline provably fails).
        legal = blanks = grid = None
        val = 0
        for _ in range(400):
            legc = tuple(cells[int(r.integers(len(cells)))])
            v = int(full[legc])
            others = [tuple(c) for c in np.array(cells)[
                r.choice(len(cells), 2, replace=False)]]
            bset = list({legc, *others})
            if len(bset) < 3 or legc not in bset:
                continue
            g = full.astype(float).copy()
            for (i, j) in bset:
                g[i, j] = 0.0
            ls = legal_set(g, bset, v)
            if ls == [legc]:
                order = list(bset)
                r.shuffle(order)
                if order[0] == legc:                  # ensure legal != first
                    order[0], order[-1] = order[-1], order[0]
                legal, blanks, grid, val = legc, order, g, v
                break
        if legal is None:                              # fallback (rare)
            legal = tuple(cells[0])
            val = int(full[legal])
            blanks = [tuple(cells[1]), legal, tuple(cells[2])]
            grid = full.astype(float).copy()
            for (i, j) in blanks:
                grid[i, j] = 0.0
        gx = lambda j: -0.10 + 0.10 * j  # noqa: E731
        gy = lambda i: 0.10 + 0.07 * i   # noqa: E731
        self._cell_xy = {(i, j): np.array([gx(j), gy(i), _Z])
                         for i in range(n) for j in range(n)}
        self.obj_names = ["num_block"]
        self.obj_pos = {"num_block": np.array([0.0, -0.12, _Z])}
        self.obj_mass = {"num_block": 0.1}
        self._legal = legal
        self._val = val
        self._grid = grid
        self.instance = {
            "grid": grid.tolist(), "value": val,
            "legal_cell": legal, "blanks": blanks,
            "description": (f"place the block with number {val} so that no "
                            "number repeats in any row or column"),
        }

    def _task_obs(self) -> dict:
        return {"grid": self._grid.flatten().astype(float),
                "value": np.array([float(self._val)])}

    def _success(self) -> bool:
        p = self.obj_pos["num_block"]
        tgt = self._cell_xy[self._legal]
        return bool(np.linalg.norm(p[:2] - tgt[:2]) < 0.045
                    and self.grasped != "num_block")

    def oracle_waypoints(self):
        s = self.obj_pos["num_block"]
        t = self._cell_xy[self._legal]
        return [
            [s[0], s[1], s[2] + 0.06, -1.0],
            [s[0], s[1], s[2], 1.0],
            [s[0], s[1], s[2] + 0.10, 1.0],
            [t[0], t[1], t[2] + 0.10, 1.0],
            [t[0], t[1], t[2], 1.0],
            [t[0], t[1], t[2], -1.0],
        ]

    def naive_waypoints(self):
        """Non-reasoning baseline: drop the block in the first blank cell
        (ignores the Latin constraint) -> usually illegal -> fails."""
        s = self.obj_pos["num_block"]
        bl = self.instance["blanks"][0]
        t = self._cell_xy[tuple(bl)]
        return [
            [s[0], s[1], s[2] + 0.06, -1.0], [s[0], s[1], s[2], 1.0],
            [s[0], s[1], s[2] + 0.10, 1.0],
            [t[0], t[1], t[2] + 0.10, 1.0], [t[0], t[1], t[2], 1.0],
            [t[0], t[1], t[2], -1.0],
        ]

class NumberBlockTask(ReasoningTask):
    category = "geometric"
    family = "number_block"
    sim_cls = NumberBlockSim
