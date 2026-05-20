"""Closed-form scorer matching the paper formulas in Sec. 3.2-3.5.

Use this when you need bit-for-bit fidelity to the equations as written;
:class:`TrajectoryEvaluator` is the streaming counterpart that produces
the same numbers online as a rollout progresses.
"""

from __future__ import annotations

import numpy as np

from .dtw import phase_completeness, unified_progress
from .safety import safety_score
from .sparc import sparc_score


def spatial_efficiency(p_unified: float, expert_len: float,
                       policy_len: float, eps: float = 1e-6,
                       gamma_expert: float = 1.0) -> float:
    """Eq. 3/8: min(gamma_expert, P_unified * L_expert / L_policy)."""
    val = (p_unified * expert_len) / (policy_len + eps)
    return float(min(gamma_expert, max(0.0, val)))


def temporal_efficiency(p_unified: float, expert_len: float, t_exec: float,
                        v_expert: float, eps: float = 1e-6,
                        gamma_expert: float = 1.0) -> float:
    """Eq. 4/9: min(gamma_expert, P_unified * L_expert / (t_exec * v_expert))."""
    val = (p_unified * expert_len) / ((t_exec + eps) * (v_expert + eps))
    return float(min(gamma_expert, max(0.0, val)))


def score_episode(policy_ee_path: np.ndarray,
                  expert_phases: list[np.ndarray],
                  phase_weights: list[float] | None,
                  speed_profile: np.ndarray,
                  forces: np.ndarray,
                  *,
                  expert_total_len: float,
                  t_exec: float,
                  v_expert: float,
                  fs: float = 20.0,
                  success: bool = False,
                  workspace: tuple[np.ndarray, np.ndarray] | None = None,
                  num_bad_contacts: int = 0,
                  gamma_expert: float = 1.0,
                  f_sus_lim: float = 40.0,
                  f_peak_lim: float = 60.0,
                  weights=(0.45, 0.15, 0.10, 0.15, 0.15)) -> dict:
    """Compute the five dimensions plus the aggregated total."""
    policy_ee_path = np.asarray(policy_ee_path, dtype=float)

    phase_scores = [phase_completeness(policy_ee_path, ph)
                    for ph in expert_phases] or [0.0]
    p_unified = 1.0 if success else unified_progress(phase_scores, phase_weights)
    score_comp = p_unified * 100.0

    diffs = np.linalg.norm(np.diff(policy_ee_path, axis=0), axis=1) \
        if len(policy_ee_path) > 1 else np.array([0.0])
    policy_len = float(np.sum(diffs))
    eta_space = spatial_efficiency(p_unified, expert_total_len, policy_len,
                                   gamma_expert=gamma_expert)
    eta_time = temporal_efficiency(p_unified, expert_total_len, t_exec,
                                   v_expert, gamma_expert=gamma_expert)
    score_space = 100.0 * eta_space
    score_time = 100.0 * eta_time

    score_smooth = sparc_score(speed_profile, fs=fs)

    saf = safety_score(forces, ee_positions=policy_ee_path,
                       workspace=workspace,
                       num_bad_contacts=num_bad_contacts,
                       num_steps=len(policy_ee_path),
                       f_sus_lim=f_sus_lim, f_peak_lim=f_peak_lim)
    score_safety = saf["safety"]

    w = weights
    total = (score_comp * w[0] + score_space * w[1] + score_time * w[2]
             + score_smooth * w[3] + score_safety * w[4])
    if success:
        total = max(total, 60.0)

    return {
        "total_fwdbias": round(total, 2),
        "total_hybrid": round(total, 2),
        "completion": round(score_comp, 2),
        "space_eff": round(float(score_space), 2),
        "time_eff": round(float(score_time), 2),
        "smoothness": round(float(score_smooth), 2),
        "safety": round(float(score_safety), 2),
        "raw_dev": 0.0,
        "raw_space_val": round(float(eta_space), 3),
        "raw_time_val": round(float(eta_time), 3),
        "raw_fft_val": 0.0,
        "p_kin": saf["p_kin"], "p_sem": saf["p_sem"], "p_dyn": saf["p_dyn"],
        "dropped": bool(phase_scores[0] > 0.9
                        and (phase_scores[-1] if len(phase_scores) > 1
                             else 1.0) < 0.5),
    }
