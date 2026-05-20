"""Dynamic Time Warping completeness (paper §3.2, equations 1 and 2).

The streaming evaluator uses a windowed-argmin alignment; this module
provides the batch DTW formulation, computing completeness as §3.2 states.

Phase completeness, eq (1)::

    S^(m)(t) = 1 - d_DTW(Policy_{0:t}, Expert^(m)) / d_DTW(Start, Expert^(m))

Sequential aggregation, eq (2): completed phases contribute their full
weight and the current phase contributes its partial progress::

    P_unified(t) = sum_{i=1..m-1} w_i  +  w_m * S^(m)(t),     sum_i w_i = 1

Eq (1) is clamped to ``[0, 1]`` with an ``eps`` denominator guard, and
eq (2)'s "completed previous phases" precondition is enforced with a
prerequisite gate ``K_active`` so a later phase only counts once its
predecessors are complete.
"""

from __future__ import annotations

import numpy as np

def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Classic O(N*M) DTW distance between two 3-D (or k-D) trajectories.

    Parameters
    ----------
    a, b:
        ``(Na, k)`` and ``(Nb, k)`` arrays of waypoints.

    Returns
    -------
    float
        The accumulated optimal-alignment cost (Euclidean local cost).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return float("inf")

    # cost matrix via broadcasting, then DP accumulation
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    acc = np.full((na + 1, nb + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, na + 1):
        ci = cost[i - 1]
        row_prev = acc[i - 1]
        row = acc[i]
        for j in range(1, nb + 1):
            row[j] = ci[j - 1] + min(row_prev[j], row[j - 1], row_prev[j - 1])
    return float(acc[na, nb])

def phase_completeness(policy_path: np.ndarray, expert_phase: np.ndarray,
                       eps: float = 1e-6) -> float:
    """Equation (1): normalised DTW completeness for one phase.

    ``S^(m) = max(0, 1 - d_DTW(policy, expert_m) / (d_DTW(start, expert_m)+eps))``.
    The clamp and ``eps`` guard are numerical stabilisers; the paper
    states the score lies in ``[0, 1]`` with 1 meaning full completion.
    """
    expert_phase = np.asarray(expert_phase, dtype=float)
    if len(expert_phase) == 0:
        return 0.0
    start = np.repeat(expert_phase[:1], 1, axis=0)
    d_policy = dtw_distance(policy_path, expert_phase)
    d_start = dtw_distance(start, expert_phase)
    return float(max(0.0, 1.0 - d_policy / (d_start + eps)))

def sequence_gate(phase_scores: list[float], threshold: float = 0.95) -> list[int]:
    """Prerequisite gate ``K_active`` (implementing eq (2)'s
    "assuming the policy has completed previous phases" precondition): 1 for
    phase ``m`` only if every prerequisite phase ``< m`` is structurally
    complete (``>= threshold``)."""
    gate, prereq_ok = [], True
    for s in phase_scores:
        gate.append(1 if prereq_ok else 0)
        prereq_ok = prereq_ok and (s >= threshold)
    return gate

def unified_progress(phase_scores: list[float], weights: list[float] | None = None,
                     gate_threshold: float = 0.95) -> float:
    """Equation (2): sequence-gated weighted aggregation, ``sum_m w_m = 1``
    (the gate enforces eq (2)'s completed-prerequisite precondition)."""
    m = len(phase_scores)
    if m == 0:
        return 0.0
    if weights is None:
        weights = [1.0 / m] * m
    w = np.asarray(weights, dtype=float)
    w = w / (w.sum() + 1e-12)
    gate = sequence_gate(phase_scores, gate_threshold)
    return float(np.sum(w * np.asarray(phase_scores) * np.asarray(gate)))
