"""End-to-end tests for augmentation, data generation, and analysis."""

import importlib

import numpy as np
import pytest

# Expert-trajectory generation and on-disk schema.
def test_expert_dataset_generation_and_schema(tmp_path):
    from robogym.data import GenConfig, generate_task_demos, write_task_dataset
    from robogym.tasks.registry import build_reasoning_suite

    task = build_reasoning_suite(seed=0)[0]
    cfg = GenConfig(demos_per_task=3, jitter=0.01)
    demos = generate_task_demos(task, cfg)
    assert len(demos) == 3
    assert all(d.success for d in demos)
    assert all(d.actions.shape[1] == 7 for d in demos)
    info = write_task_dataset(tmp_path, task.name, demos)
    assert info["num_demos"] == 3
    import glob
    f = glob.glob(str(tmp_path / "*_demo.*"))[0]
    if f.endswith(".npz"):
        d = np.load(f)
        assert "demo_0/actions" in d and "demo_0/states" in d

def test_dataset_is_domain_balanced():
    from robogym.data import GenConfig, generate_dataset

    rep = None
    for _, _, rep in generate_dataset(GenConfig(demos_per_task=1)):
        pass
    assert rep is not None
    assert set(rep.per_category) == {"geometric", "physical", "memory"}
    assert rep.total_demos == 50


def test_model_registry_covers_all_paper_models():
    from paper_results import ALL_MODELS
    from robogym.policies import MODEL_REGISTRY, make_mock_fleet

    for m in ALL_MODELS:
        assert m in MODEL_REGISTRY, m
    assert len(make_mock_fleet()) == 8

def test_pi05_runner_importable():
    m = importlib.import_module("robogym.runners.run_libero_eval_pi05")
    assert hasattr(m, "eval_pi05") and hasattr(m, "Pi05EvalConfig")
    assert m.Pi05EvalConfig().task_suite_name == "libero_object"


def test_code_agent_synthesizes_and_repairs(tmp_path):
    from robogym.augmentation import CodeRepairLoop

    rec = {"phase_weights": [0.4, 0.6], "expert_total_len": 1.0,
           "f_sus_lim": 40.0, "f_peak_lim": 60.0, "gamma_expert": 1.0,
           "num_phases": 2, "task_description": "t"}
    loop = CodeRepairLoop(out_dir=tmp_path, max_rounds=3)
    ok = loop.run(rec)
    assert ok.valid and ok.rounds == 0 and ok.recovery == "first_try"
    assert {"completion", "safety", "total_fwdbias"} <= set(ok.sample_scores)
    fixed = loop.run(rec, broken_first=True)
    assert fixed.valid and fixed.rounds >= 1
    assert fixed.recovery == "baseline_fallback"


def test_asset_library_builds_with_controlled_physics(tmp_path):
    from robogym.assets import build_asset_library

    man = build_asset_library(tmp_path, seed=1, episodes=1)
    assert man["assets"]
    a = man["assets"][0]
    for k in ("obj_path", "tex_path", "mjcf_path", "physics"):
        assert k in a
    assert 0.0 < a["physics"]["mass"] < 2.0
    assert 0.0 < a["physics"]["friction"] < 1.5

def test_all_augmented_libero_suites():
    from robogym.runners.suites import LIBERO_SUITE_SIZES, SyntheticSuite

    for name, size in LIBERO_SUITE_SIZES.items():
        s = SyntheticSuite(name, seed=0)
        assert s.n_tasks == size, (name, s.n_tasks, size)
        assert s.task_spec(0).task_description

def test_human_study_alignment_matches_paper_shape():
    from paper_results import TABLE3_REASONING
    from robogym.analysis import HumanStudy

    study = HumanStudy(models=list(TABLE3_REASONING))
    study.simulate_experts(TABLE3_REASONING, n_experts=3, noise=0.3, seed=0)
    W = study.inter_rater_W()
    assert 0.0 <= W["overall"] <= 1.0
    rho = study.spearman_vs_automated(TABLE3_REASONING)
    assert -1.0 <= rho["mean"] <= 1.0 and len(rho) == 6

@pytest.mark.skipif(
    importlib.util.find_spec("libero") is None,
    reason="real LIBERO stack not installed")
def test_real_libero_backend_integration():  # pragma: no cover
    from robogym.runners.suites import LiberoSuite

    suite = LiberoSuite("libero_object", datasets_root="./LIBERO/datasets")
    assert suite.n_tasks == 10
    env = suite.make_env(0)
    assert env.task_description
