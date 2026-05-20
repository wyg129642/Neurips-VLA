"""DTW phase completeness (Sec. 3.2, Eq. 1-2 and Eq. 6-7)."""

from __future__ import annotations

import numpy as np


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean DTW between two waypoint sequences."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return float("inf")

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
    """Eq. 1/6: normalized DTW completeness with zero-truncation."""
    expert_phase = np.asarray(expert_phase, dtype=float)
    if len(expert_phase) == 0:
        return 0.0
    start = np.repeat(expert_phase[:1], 1, axis=0)
    d_policy = dtw_distance(policy_path, expert_phase)
    d_start = dtw_distance(start, expert_phase)
    return float(max(0.0, 1.0 - d_policy / (d_start + eps)))


def sequence_gate(phase_scores: list[float], threshold: float = 0.95
                  ) -> list[int]:
    """Indicator function K_active from Eq. 7: 1 only after prerequisites pass."""
    gate, prereq_ok = [], True
    for s in phase_scores:
        gate.append(1 if prereq_ok else 0)
        prereq_ok = prereq_ok and (s >= threshold)
    return gate


def unified_progress(phase_scores: list[float],
                     weights: list[float] | None = None,
                     gate_threshold: float = 0.95) -> float:
    """Eq. 2/7: gated weighted aggregation, sum_m w_m = 1."""
    m = len(phase_scores)
    if m == 0:
        return 0.0
    if weights is None:
        weights = [1.0 / m] * m
    w = np.asarray(weights, dtype=float)
    w = w / (w.sum() + 1e-12)
    gate = sequence_gate(phase_scores, gate_threshold)
    return float(np.sum(w * np.asarray(phase_scores) * np.asarray(gate)))
