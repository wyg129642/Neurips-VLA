"""Trajectory evaluator and canonical metric tests."""

import numpy as np

from robogym.envs import SyntheticPickPlaceSim
from robogym.metrics import TrajectoryEvaluator, score_episode
from robogym.metrics.dtw import phase_completeness, unified_progress
from robogym.metrics.safety import dynamic_penalty, safety_score
from robogym.metrics.sparc import sparc, sparc_score


def _eval_on(actions, init_state, policy_actions, dt=0.05):
    sim = SyntheticPickPlaceSim(task_objects=["alphabet_soup"], seed=0, dt=dt)
    ev = TrajectoryEvaluator(sim, actions, init_state, dt=dt)
    sim.set_state(init_state); sim.forward()
    for t, a in enumerate(policy_actions):
        _, _, _, info = sim.step(np.asarray(a).tolist())
        ev.update(np.asarray(a), info, step_idx=t)
    return ev.calculate_score()


def test_oracle_replay_scores_high():
    sim = SyntheticPickPlaceSim(task_objects=["alphabet_soup"], seed=0)
    ea, eis = sim.generate_expert_demo()
    s = _eval_on(ea, eis, ea)
    assert s["completion"] >= 70.0, s
    assert s["total_fwdbias"] >= 55.0, s
    for k in ("space_eff", "time_eff", "smoothness", "safety"):
        assert 0.0 <= s[k] <= 100.0
    assert s["smoothness"] >= 50.0, s
    assert s["safety"] >= 80.0, s


def test_idle_policy_scores_low():
    sim = SyntheticPickPlaceSim(task_objects=["alphabet_soup"], seed=0)
    ea, eis = sim.generate_expert_demo()
    s = _eval_on(ea, eis, [np.zeros(7)] * len(ea))
    assert s["completion"] < 60.0, s


def test_score_schema_matches_csv():
    sim = SyntheticPickPlaceSim(task_objects=["milk"], seed=1)
    ea, eis = sim.generate_expert_demo()
    s = _eval_on(ea, eis, ea)
    for col in ("total_fwdbias", "total_hybrid", "completion", "space_eff",
                "time_eff", "smoothness", "safety", "raw_dev",
                "raw_space_val", "raw_time_val", "raw_fft_val", "dropped"):
        assert col in s


def test_dtw_completeness_bounds():
    expert = np.cumsum(np.ones((30, 3)) * 0.05, axis=0)
    assert phase_completeness(expert, expert) > 0.95
    assert phase_completeness(expert[:1], expert) < 0.30
    assert 0.0 <= unified_progress([0.5, 0.0], [0.5, 0.5]) <= 1.0


def test_sequence_gate_blocks_later_phases():
    assert unified_progress([0.4, 0.9], [0.5, 0.5]) < \
        unified_progress([1.0, 0.9], [0.5, 0.5])


def test_sparc_smoother_is_closer_to_zero():
    t = np.linspace(0, 1, 200)
    smooth = np.sin(2 * np.pi * t)
    jittery = smooth + 0.4 * np.random.default_rng(0).standard_normal(200)
    assert abs(sparc(smooth)) < abs(sparc(jittery))
    assert sparc_score(smooth) > sparc_score(jittery)


def test_dynamic_safety_penalty_monotonic():
    safe = np.full(100, 10.0)
    unsafe = np.concatenate([np.full(95, 10.0), np.full(5, 200.0)])
    assert dynamic_penalty(safe) < dynamic_penalty(unsafe)
    assert safety_score(safe)["safety"] > safety_score(unsafe)["safety"]


def test_paper_scorer_drop_in():
    sim = SyntheticPickPlaceSim(task_objects=["butter"], seed=2)
    ea, eis = sim.generate_expert_demo()
    ev = TrajectoryEvaluator(sim, ea, eis)
    phases = [ev.expert_ee_path[: ev.reach_cutoff_idx + 1]] + \
        [o["path"] for o in ev.tracked_objects]
    s = score_episode(ev.expert_ee_path, phases, None,
                      np.ones(50) * 0.2, np.ones(50) * 5.0,
                      expert_total_len=float(ev.ee_total_len),
                      t_exec=2.0, v_expert=0.3, success=True)
    assert s["completion"] == 100.0 and s["total_fwdbias"] >= 60.0
