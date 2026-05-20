"""Paper-faithful scorer. §3.2-3.5 (equations 1-5 + qualitative §3.5).

method is §3.2 eq (1)/(2) completeness, §3.3 eq (3)/(4) spatial/temporal
efficiency, §3.4 eq (5) SPARC smoothness, and §3.5 (qualitative) safety. This
is the canonical counterpart to the heuristic
:class:`TrajectoryEvaluator`: given a recorded episode (policy EE path, expert
phase paths, speed profile, forces) it computes the five dimensions following
§3.2-3.5. Use this when you want results that match the written method; use the streaming evaluator for bit-for-bit parity with the metric
engine. Both are wired into the runner via ``--scorer``.
"""

from __future__ import annotations

import numpy as np

from .dtw import phase_completeness, unified_progress
from .safety import safety_score
from .sparc import sparc_score

def spatial_efficiency(p_unified: float, expert_len: float,
                       policy_len: float, eps: float = 1e-6,
                       gamma_expert: float = 1.0) -> float:
    """Equation (3): ``η_space = min(γ, P_unified * L_expert / L_policy)``.

    §3.3 caps η_space "at a maximum of 1", so ``gamma_expert`` defaults to 1.0
    (the paper's cap). Allowing ``gamma_expert > 1`` is an optional extension
    (optional expert-sub-optimality slack).
    """
    val = (p_unified * expert_len) / (policy_len + eps)
    return float(min(gamma_expert, val))

def temporal_efficiency(p_unified: float, expert_len: float, t_exec: float,
                         v_expert: float, eps: float = 1e-6,
                         gamma_expert: float = 1.0) -> float:
    """Equation (4): ``η_time = min(γ, P_unified*L_expert/((t_exec+eps)*v_expert))``.
    §3.3 caps η_time at 1 (``gamma_expert`` default 1.0)."""
    val = (p_unified * expert_len) / ((t_exec + eps) * (v_expert + eps))
    return float(min(gamma_expert, val))

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
                   weights=(0.45, 0.15, 0.10, 0.15, 0.15)) -> dict:
    """Compute the paper's five dimensions + the weighted total.

    Returns the same key schema as
    :meth:`TrajectoryEvaluator.calculate_score` so it is a drop-in alternative
    for the runner / CSV aggregation.
    """
    policy_ee_path = np.asarray(policy_ee_path, dtype=float)

    # Completeness (§3.2 eq 1,2)
    phase_scores = [phase_completeness(policy_ee_path, ph)
                    for ph in expert_phases] or [0.0]
    p_unified = 1.0 if success else unified_progress(phase_scores, phase_weights)
    score_comp = p_unified * 100.0

    # Spatial / Temporal efficiency (§3.3 eq 3,4)
    diffs = np.linalg.norm(np.diff(policy_ee_path, axis=0), axis=1) \
        if len(policy_ee_path) > 1 else np.array([0.0])
    policy_len = float(np.sum(diffs))
    raw_space = spatial_efficiency(p_unified, expert_total_len, policy_len,
                                   gamma_expert=gamma_expert)
    raw_time = temporal_efficiency(p_unified, expert_total_len, t_exec,
                                   v_expert, gamma_expert=gamma_expert)
    # calibrate to [0,100] with the sigmoids for cross-comparability
    # ([0,100] mapping for consistent scaling)
    score_space = 100.0 / (1.0 + np.exp(-4.0 * (raw_space - 0.55)))
    score_time = 100.0 / (1.0 + np.exp(-25.0 * (raw_time - 0.15)))

    # Smoothness (§3.4 eq 5 SPARC)
    score_smooth = sparc_score(speed_profile, fs=fs)

    # Safety (§3.5)
    saf = safety_score(forces, ee_positions=policy_ee_path,
                        workspace=workspace,
                        num_bad_contacts=num_bad_contacts,
                        num_steps=len(policy_ee_path))
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
        "raw_space_val": round(float(raw_space), 3),
        "raw_time_val": round(float(raw_time), 3),
        "raw_fft_val": 0.0,
        "p_kin": saf["p_kin"], "p_sem": saf["p_sem"], "p_dyn": saf["p_dyn"],
        "dropped": bool(phase_scores[0] > 0.9
                        and (phase_scores[-1] if len(phase_scores) > 1
                             else 1.0) < 0.5),
    }
