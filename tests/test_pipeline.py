"""End-to-end pipeline tests: runner -> CSV tree -> analysis."""

from pathlib import Path

from robogym.analysis import (
    kendalls_w,
    metric_correlation_matrix,
    reproduce_paper_figures,
    reproduce_paper_tables,
    runs_to_model_table,
)
from robogym.augmentation import AugmentationPipeline
from robogym.metrics.aggregate import (
    GLOBAL_HEADER,
    SUITE_HEADER,
    TASK_DETAIL_HEADER,
)
from robogym.policies import MockPolicy, OraclePolicy
from robogym.runners import EvalConfig, run_suite
from robogym.runners.suites import SyntheticSuite

def test_runner_produces_csv_tree(tmp_path):
    cfg = EvalConfig(backend="synthetic", task_suite_name="libero_object",
                     num_tasks=2, num_trials_per_task=2, max_steps=140,
                     results_dir=str(tmp_path), run_id_note="t")
    g = run_suite(cfg, OraclePolicy())
    rd = tmp_path / "EVAL-CUSTOM-libero_object-oracle--t"
    assert (rd / "global_summary.csv").exists()
    assert (rd / "suite_summary.csv").exists()
    tds = list(rd.glob("libero_object/task_*/task_details.csv"))
    assert len(tds) == 2
    # schema parity with the OpenVLA-OFT runner
    header = (rd / "global_summary.csv").read_text().splitlines()[0]
    assert header == ",".join(GLOBAL_HEADER)
    assert (rd / "suite_summary.csv").read_text().splitlines()[0] == \
        ",".join(SUITE_HEADER)
    assert tds[0].read_text().splitlines()[0] == ",".join(TASK_DETAIL_HEADER)
    assert 0.0 <= g["success_rate"] <= 1.0

def test_oracle_beats_weak_mock(tmp_path):
    common = dict(backend="synthetic", task_suite_name="libero_object",
                  num_tasks=3, num_trials_per_task=1, max_steps=150,
                  results_dir=str(tmp_path))
    go = run_suite(EvalConfig(**common, run_id_note="o"), OraclePolicy())
    gm = run_suite(EvalConfig(**common, run_id_note="m"),
                   MockPolicy(competence=0.05, seed=1, name="weak"))
    assert go["avg_total_fwdbias"] > gm["avg_total_fwdbias"]

def test_augmentation_pipeline_record(tmp_path):
    suite = SyntheticSuite("libero_object", seed=3)
    env = suite.make_env(0)
    ea, eis = suite.expert_demo(env, 0)
    aug = AugmentationPipeline().augment(env.backend, ea, eis,
                                         suite.task_spec(0).task_description)
    rec = aug.augmentation
    assert abs(sum(rec["phase_weights"]) - 1.0) < 1e-6
    assert rec["expert_total_len"] > 0 and rec["f_peak_lim"] >= rec["f_sus_lim"]

def test_reasoning_backend_runs(tmp_path):
    cfg = EvalConfig(backend="reasoning",
                     task_suite_name="system2_reasoning",
                     num_tasks=4, num_trials_per_task=1, max_steps=160,
                     results_dir=str(tmp_path), run_id_note="r")
    g = run_suite(cfg, OraclePolicy())
    assert g["num_tasks"] == 4

def test_analysis_reproduces_paper_artifacts(tmp_path):
    tabs = reproduce_paper_tables(tmp_path / "tab")
    figs = reproduce_paper_figures(tmp_path / "fig")
    assert tabs["report"].exists() and tabs["table2"].exists()
    assert len(figs) >= 6 and all(Path(f).exists() for f in figs)

def test_correlation_helpers():
    from paper_results import paper_tables as P
    M = metric_correlation_matrix(P.TABLE3_REASONING)
    assert M.shape == (5, 5)
    assert abs(M[0, 0] - 1.0) < 1e-9
    import numpy as np
    w = kendalls_w(np.array([[1, 2, 3, 4], [1, 2, 3, 4], [2, 1, 3, 4]]))
    assert 0.0 <= w <= 1.0
