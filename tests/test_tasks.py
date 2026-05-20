"""System-2 reasoning task-suite tests."""

import numpy as np

from robogym.tasks import (
    ColorHanoiTask,
    MazeTask,
    NumberBlockTask,
    SeesawWeightTask,
    SequentialCountingTask,
    TangramTask,
    build_reasoning_suite,
    reasoning_suite_by_category,
)


def test_suite_has_50_tasks_in_3_levels():
    suite = build_reasoning_suite(seed=0)
    assert len(suite) == 50
    by = reasoning_suite_by_category(0)
    assert set(by) == {"geometric", "physical", "memory"}
    assert sum(len(v) for v in by.values()) == 50


def test_test_time_randomization_changes_instance():
    a = MazeTask(seed=1)._instance
    b = MazeTask(seed=2)._instance
    assert not np.allclose(a["goal"], b["goal"]) or a["mirror"] != b["mirror"]


def test_every_task_yields_solvable_expert_demo():
    for cls in (MazeTask, SeesawWeightTask, TangramTask, NumberBlockTask,
                ColorHanoiTask, SequentialCountingTask):
        t = cls(seed=7)
        actions, init_state = t.expert_demo(seed=7)
        assert actions.ndim == 2 and actions.shape[1] == 7
        assert len(actions) > 5
        assert init_state.size > 0


def test_maze_progress_logic():
    """Detour progress increases start -> mid -> target; off-table zeroes out."""
    t = MazeTask(seed=0)
    inst = t._instance
    p0 = MazeTask.get_progress(np.asarray(inst["start"], float), inst)
    p1 = MazeTask.get_progress(np.asarray(inst["mid"], float), inst)
    p2 = MazeTask.get_progress(np.asarray(inst["goal"], float), inst)
    assert 0.0 <= p0 < p1 < p2 <= 1.0
    fell = np.array([inst["goal"][0], inst["goal"][1], 0.0])
    assert MazeTask.get_progress(fell, inst) == 0.0


def test_reasoning_tasks_genuinely_require_reasoning():
    """Oracle should solve, non-reasoning baseline should mostly fail."""
    import collections

    from robogym.tasks import build_reasoning_suite

    agg = collections.defaultdict(lambda: [0, 0, 0])
    for s in range(4):
        for task in build_reasoning_suite(seed=s):
            sim = task.make_sim()
            ea, eis = sim.generate_expert_demo(seed=s * 13 + task.task_id)
            sim.set_state(eis); sim.forward()
            o = any(sim.step(np.asarray(a).tolist())[3]["success"]
                    for a in ea)
            na, nis = sim.generate_naive_demo()
            sim.set_state(nis); sim.forward()
            nv = any(sim.step(np.asarray(a).tolist())[3]["success"]
                     for a in na)
            r = agg[task.family]
            r[0] += 1; r[1] += int(o); r[2] += int(nv)
    for fam, (n, o, nv) in agg.items():
        assert o / n >= 0.9, f"{fam} oracle SR too low: {o}/{n}"
        assert nv / n <= 0.65, f"{fam} not reasoning-gated: naive {nv}/{n}"


def test_reasoning_task_env_is_runnable():
    from robogym.metrics import TrajectoryEvaluator
    t = TangramTask(seed=3)
    env = t.make_env()
    ea, eis = t.expert_demo(seed=3)
    ev = TrajectoryEvaluator(env.backend, ea, eis)
    env.backend.set_state(eis)
    env.backend.forward()
    for i, a in enumerate(ea):
        _, _, _, info = env.step(np.asarray(a).tolist())
        ev.update(np.asarray(a), info, step_idx=i)
    s = ev.calculate_score()
    assert 0.0 <= s["total_fwdbias"] <= 100.0
