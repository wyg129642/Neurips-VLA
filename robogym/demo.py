"""End-to-end RoboGym demo.

Runs on the synthetic backend so the pipeline can be exercised without a
GPU, model weights, or MuJoCo:

    python -m robogym.demo                 # full demo
    python -m robogym.demo --quick         # tiny smoke run
    python -m robogym.demo --out results   # choose output dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .analysis import render_paper_figures, render_paper_tables
from .analysis.figures import metric_bars, radar_chart
from .analysis.loader import runs_to_model_table
from .augmentation import AugmentationPipeline
from .config import PAPER_MODEL_COMPETENCE
from .policies import MockPolicy, OraclePolicy
from .runners import EvalConfig, run_suite
from .runners.suites import SyntheticSuite


def _banner(t: str) -> None:
    print("\n" + "=" * 72 + f"\n  {t}\n" + "=" * 72)


def oracle_sanity(out: Path, quick: bool) -> None:
    _banner("1. Oracle sanity check (expert replay should score near ceiling)")
    cfg = EvalConfig(backend="synthetic", scorer="streaming",
                     task_suite_name="libero_object",
                     num_tasks=2 if quick else 3,
                     num_trials_per_task=1 if quick else 2,
                     max_steps=160, results_dir=str(out / "raw"),
                     run_id_note="oracle")
    g = run_suite(cfg, OraclePolicy())
    print(f"  oracle suite avg total = {g.get('avg_total_fwdbias', 0):.2f}")


def augmentation_demo(quick: bool) -> None:
    _banner("2. Task-augmentation pipeline")
    suite = SyntheticSuite("libero_object", seed=1)
    pipe = AugmentationPipeline()
    env = suite.make_env(0)
    ea, eis = suite.expert_demo(env, 0)
    aug = pipe.augment(env.backend, ea, eis,
                       suite.task_spec(0).task_description)
    rec = aug.augmentation
    print(f"  task: {rec['task_description']}")
    print(f"  tracked moved objects : {rec['tracked_objects']}")
    print("  synthesised phase wts : "
          f"{[round(w, 3) for w in rec['phase_weights']]}")
    print(f"  safe force limits     : sustained={rec['f_sus_lim']:.1f} N  "
          f"peak={rec['f_peak_lim']:.1f} N")
    print(f"  synthesiser           : {rec['synthesizer']}")


def models_on_augmented_libero(out: Path, quick: bool) -> dict[str, str]:
    _banner("3. Augmented LIBERO on surrogate policies")
    run_dirs = {}
    for model, comp in PAPER_MODEL_COMPETENCE.items():
        pol = MockPolicy(competence=comp["libero"], seed=hash(model) % 9999,
                         drop_prob=0.10, name=model)
        cfg = EvalConfig(backend="synthetic", scorer="streaming",
                         task_suite_name="libero_object",
                         num_tasks=3 if quick else 6,
                         num_trials_per_task=1 if quick else 2,
                         max_steps=160, results_dir=str(out / "raw"),
                         run_id_note="auglibero")
        run_suite(cfg, pol)
        run_dirs[model] = (out / "raw"
                           / f"EVAL-CUSTOM-libero_object-{model}--auglibero")
    return run_dirs


def models_on_reasoning(out: Path, quick: bool) -> dict[str, str]:
    _banner("4. System-2 reasoning suite on surrogate policies (subset)")
    run_dirs = {}
    top = list(PAPER_MODEL_COMPETENCE)[:3 if quick else 4]
    for model in top:
        comp = PAPER_MODEL_COMPETENCE[model]
        pol = MockPolicy(competence=comp["reasoning"],
                         seed=hash(model) % 9999, drop_prob=0.35, name=model)
        cfg = EvalConfig(backend="reasoning", scorer="streaming",
                         task_suite_name="system2_reasoning",
                         num_tasks=6 if quick else 12,
                         num_trials_per_task=1,
                         max_steps=200, results_dir=str(out / "raw"),
                         run_id_note="reasoning")
        run_suite(cfg, pol)
        run_dirs[model] = (out / "raw"
                           / f"EVAL-CUSTOM-system2_reasoning-{model}--reasoning")
    return run_dirs


def analysis(out: Path, libero_runs: dict, reason_runs: dict) -> None:
    _banner("5. Render paper tables and figures")
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    live = runs_to_model_table(libero_runs)
    if live:
        radar_chart(live, fig_dir / "live_radar_augmented_libero.png",
                    "Augmented LIBERO (surrogate-policy run)")
        metric_bars(live, fig_dir / "live_bars_augmented_libero.png",
                    "Augmented LIBERO (surrogate-policy run)")
        print(f"  live surrogate-run table ({len(live)} models) "
              f"written to figures/live_*")

    tabs = render_paper_tables(out / "paper_tables")
    figs = render_paper_figures(fig_dir)
    print(f"  paper tables  : {tabs['report']}")
    print(f"  paper figures : {len(figs)} files in {fig_dir}")


def fidelity_showcase(out: Path, quick: bool) -> None:
    """Exercise the reasoning gate, dataset pipeline, code agent, assets,
    and the human-study harness."""
    import collections

    from robogym.analysis import HumanStudy
    from robogym.assets import build_asset_library
    from robogym.augmentation import CodeRepairLoop
    from robogym.data import GenConfig, generate_task_demos
    from robogym.tasks.registry import build_reasoning_suite
    from paper_results import TABLE3_REASONING

    _banner("6. Subsystem showcase")

    agg = collections.defaultdict(lambda: [0, 0, 0])
    for s in range(2 if quick else 5):
        for t in build_reasoning_suite(seed=s):
            sim = t.make_sim()
            ea, eis = sim.generate_expert_demo(seed=s * 13 + t.task_id)
            sim.set_state(eis); sim.forward()
            o = any(sim.step(np.asarray(a).tolist())[3]["success"]
                    for a in ea)
            na, nis = sim.generate_naive_demo()
            sim.set_state(nis); sim.forward()
            nv = any(sim.step(np.asarray(a).tolist())[3]["success"]
                     for a in na)
            r = agg[t.category]; r[0] += 1; r[1] += o; r[2] += nv
        if quick:
            break
    print("  reasoning gate (success requires physical/logical state):")
    for c, (n, o, nv) in sorted(agg.items()):
        print(f"    {c:9s}  oracleSR={o/n:.2f}  non-reasoningSR={nv/n:.2f}")

    task = build_reasoning_suite(0)[0]
    demos = generate_task_demos(task, GenConfig(demos_per_task=3))
    print(f"  expert-demo pipeline: {len(demos)} success-filtered demos for "
          f"'{task.name}'")

    loop = CodeRepairLoop(out_dir=out / "scoring_scripts")
    rec = {"phase_weights": [0.5, 0.5], "expert_total_len": 1.0,
           "f_sus_lim": 40.0, "f_peak_lim": 60.0, "gamma_expert": 1.0,
           "num_phases": 2, "task_description": task.description}
    good = loop.run(rec)
    fixed = loop.run(rec, broken_first=True)
    print(f"  code agent: clean recovery={good.recovery}; "
          f"injected-fault recovery={fixed.recovery}")

    man = build_asset_library(out / "assets", seed=0, episodes=1)
    print(f"  asset library: {len(man['assets'])} assets")

    study = HumanStudy(models=list(TABLE3_REASONING))
    study.simulate_experts(TABLE3_REASONING, n_experts=3, seed=0)
    rep = study.report(TABLE3_REASONING, out / "human_study")
    print(f"  human-study harness (test mode): Kendall W="
          f"{rep['kendalls_w']['overall']}, mean rho vs automated="
          f"{rep['spearman_vs_automated']['mean']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="RoboGym end-to-end demo")
    ap.add_argument("--out", default="results", help="output directory")
    ap.add_argument("--quick", action="store_true", help="tiny/fast smoke run")
    args = ap.parse_args(argv)

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"RoboGym demo -> {out}  (quick={args.quick})")

    oracle_sanity(out, args.quick)
    augmentation_demo(args.quick)
    libero_runs = models_on_augmented_libero(out, args.quick)
    reason_runs = models_on_reasoning(out, args.quick)
    analysis(out, libero_runs, reason_runs)
    fidelity_showcase(out, args.quick)

    _banner("DONE")
    print(f"  All artifacts under: {out}")
    print("  - raw/             CSV tree")
    print("  - figures/         fig3a/3b are paper Figure 3 radars; diag_* "
          "are diagnostics; live_* are from the surrogate run.")
    print("  - paper_tables/    rendered Tables 2-6")
    print("  - assets/          generated reasoning-centric asset library")
    print("  - human_study/     human-alignment report (test mode)")
    print("  - scoring_scripts/ scoring scripts emitted by the code agent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
