"""Cubic B-spline interpolation used by the expert oracle (paper Sec. 4.1)."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import make_interp_spline


def bspline_sample(waypoints_xyz, samples_per_segment: int = 8,
                   degree: int = 3) -> np.ndarray:
    """Sample a clamped cubic B-spline through cartesian waypoints.

    Returns ``((n-1) * samples_per_segment, 3)`` positions, excluding the
    starting waypoint. Falls back to piecewise-linear resampling when fewer
    than ``degree + 1`` waypoints are provided.
    """
    pts = np.asarray(waypoints_xyz, float)
    n = len(pts)
    if n < degree + 1:
        return _piecewise_linear(pts, samples_per_segment)
    u_wp = np.arange(n, dtype=float)
    spl = make_interp_spline(u_wp, pts, k=degree, bc_type="clamped")
    n_total = (n - 1) * samples_per_segment
    u = np.linspace(0.0, n - 1, n_total + 1)[1:]
    return np.asarray(spl(u), float)


def grip_schedule(n_waypoints: int, samples_per_segment: int,
                  grips_at_wp) -> np.ndarray:
    """Per-sample gripper command, inherited from each segment's destination."""
    grips_at_wp = np.asarray(grips_at_wp, float)
    n_total = (n_waypoints - 1) * samples_per_segment
    u = np.linspace(0.0, n_waypoints - 1, n_total + 1)[1:]
    idx = np.clip(np.ceil(u).astype(int), 1, n_waypoints - 1)
    return grips_at_wp[idx]


def _piecewise_linear(pts: np.ndarray, samples_per_segment: int) -> np.ndarray:
    out = []
    for i in range(len(pts) - 1):
        for k in range(1, samples_per_segment + 1):
            s = k / samples_per_segment
            out.append(pts[i] + s * (pts[i + 1] - pts[i]))
    return np.asarray(out, float) if out else pts.copy()
