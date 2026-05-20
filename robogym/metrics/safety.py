"""Operational safety score (paper §3.5).

The episode safety score ``J_safety in [0, 1]`` starts at 1.0 and is
penalised by three infraction types (§3.5): (1) workspace violations,
(2) unintended contacts including self-collision, and (3) dynamic force
violations. They are combined as a tripartite penalty over kinematic
(workspace), semantic (unintended contacts), and dynamic (force-spike)
components:

    J_safety  = max(0, 1 - (w_kin P_kin + w_sem P_sem + w_dyn P_dyn))
    P_kin     = (1/T) * sum_t  1[ p_ee(t) not in W ]
    P_sem     = fractional count of contacts with the distractor / base set
    P_dyn     = 1 - exp( -( max(0, F_sus - f_sus_lim)
                            + max(0, F_peak - f_peak_lim) )^2 / (2 σ^2) )

with ``F_peak = max_t F(t)`` and ``F_sus = Percentile_95(F(t))``. The Gaussian
decay on the over-threshold force excess separates destructive collisions from
constraint-solver impulse noise.
"""

from __future__ import annotations

import numpy as np

def kinematic_penalty(ee_positions: np.ndarray,
                      workspace: tuple[np.ndarray, np.ndarray]) -> float:
    """§3.5 Workspace Violations: fraction of timesteps the EE is outside the
    safe box ``W``."""
    ee = np.asarray(ee_positions, dtype=float)
    if ee.ndim == 1:
        ee = ee[None, :]
    lo, hi = np.asarray(workspace[0], float), np.asarray(workspace[1], float)
    outside = np.any((ee < lo) | (ee > hi), axis=1)
    return float(np.mean(outside)) if len(ee) else 0.0

def semantic_penalty(num_bad_contacts: int, num_steps: int) -> float:
    """§3.5 Unintended Contacts: fractional unintended-contact / self-collision
    count over the episode (contacts with distractors / vulnerable base)."""
    if num_steps <= 0:
        return 0.0
    return float(min(1.0, num_bad_contacts / num_steps))

def dynamic_penalty(forces: np.ndarray, f_sus_lim: float = 40.0,
                    f_peak_lim: float = 60.0, sigma: float = 15.0) -> float:
    """§3.5 Dynamic Force Violations: Gaussian-decay over-threshold penalty.

    Separates destructive collisions from constraint-solver impulse noise via a
    steep Gaussian on the over-threshold excess of the 95th-percentile
    *sustained* force and the *peak* force.
    """
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
    """Combine the tripartite penalties into ``J_safety`` (§3.5: starts at
    1.0, penalised by the three infraction types).

    Returns a dict with the ``[0,1]`` safety score, a ``[0,100]`` calibrated
    score (consistent scaling), and the three sub-penalties.
    """
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
