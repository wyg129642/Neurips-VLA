"""Tripartite operational-safety score (Sec. 3.5, Eq. 11-13)."""

from __future__ import annotations

import numpy as np


def kinematic_penalty(ee_positions: np.ndarray,
                      workspace: tuple[np.ndarray, np.ndarray]) -> float:
    """Fraction of timesteps the EE leaves the safe box ``W``."""
    ee = np.asarray(ee_positions, dtype=float)
    if ee.ndim == 1:
        ee = ee[None, :]
    if len(ee) == 0:
        return 0.0
    lo, hi = np.asarray(workspace[0], float), np.asarray(workspace[1], float)
    outside = np.any((ee < lo) | (ee > hi), axis=1)
    return float(np.mean(outside))


def semantic_penalty(num_bad_contacts: int, num_steps: int) -> float:
    """Fractional unintended-contact count over the episode."""
    if num_steps <= 0:
        return 0.0
    return float(min(1.0, num_bad_contacts / num_steps))


def dynamic_penalty(forces: np.ndarray, f_sus_lim: float = 40.0,
                    f_peak_lim: float = 60.0, sigma: float = 15.0) -> float:
    """Eq. 13: Gaussian decay on over-threshold sustained / peak force."""
    f = np.asarray(forces, dtype=float).ravel()
    if len(f) == 0:
        return 0.0
    f_sus = float(np.percentile(f, 95))
    f_peak = float(np.max(f))
    excess = max(0.0, f_sus - f_sus_lim) + max(0.0, f_peak - f_peak_lim)
    return float(1.0 - np.exp(-(excess ** 2) / (2.0 * sigma ** 2)))


def safety_score(forces: np.ndarray,
                 ee_positions: np.ndarray | None = None,
                 workspace: tuple[np.ndarray, np.ndarray] | None = None,
                 num_bad_contacts: int = 0,
                 num_steps: int | None = None,
                 w_kin: float = 0.34, w_sem: float = 0.33,
                 w_dyn: float = 0.33,
                 f_sus_lim: float = 40.0, f_peak_lim: float = 60.0,
                 sigma: float = 15.0) -> dict:
    """Combine the three penalties into ``J_safety`` (Eq. 11)."""
    p_dyn = dynamic_penalty(forces, f_sus_lim, f_peak_lim, sigma)
    p_kin = (kinematic_penalty(ee_positions, workspace)
             if ee_positions is not None and workspace is not None else 0.0)
    n = num_steps if num_steps is not None else (
        len(forces) if forces is not None else 0)
    p_sem = semantic_penalty(num_bad_contacts, n)

    j = max(0.0, 1.0 - (w_kin * p_kin + w_sem * p_sem + w_dyn * p_dyn))
    return {
        "j_safety": float(j),
        "safety": round(float(j) * 100.0, 2),
        "p_kin": round(p_kin, 4),
        "p_sem": round(p_sem, 4),
        "p_dyn": round(p_dyn, 4),
    }
