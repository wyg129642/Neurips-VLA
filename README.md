# RoboGym

Beyond Success Rates in Vision-Language-Action Models via Multi-Dimensional
Evaluation.

RoboGym is a multi-dimensional benchmark for Vision-Language-Action (VLA)
models that scores every rollout on five dimensions rather than a binary
success bit. It ships a 50-task System-2 reasoning suite with test-time
randomization of task-critical variables (object position, color, mass,
target sequence) so that solutions cannot be memorised.

## Evaluation dimensions

| Dimension              | Section      | Implementation                                  |
|------------------------|--------------|-------------------------------------------------|
| Completeness           | §3.2, Eq 1-2 | Sequence-gated DTW progress against expert demo |
| Spatial Efficiency     | §3.3, Eq 3   | `min(γ, P_unified · L_expert / L_policy)`       |
| Temporal Efficiency    | §3.3, Eq 4   | `min(γ, P_unified · L_expert / (t_exec·v̄_expert))` |
| Kinematic Smoothness   | §3.4, Eq 5   | Spectral Arc Length (SPARC) of EE velocity      |
| Operational Safety     | §3.5, Eq 11  | Tripartite workspace / contact / dynamic-force  |

The streaming evaluator (`robogym.metrics.TrajectoryEvaluator`) and the
closed-form scorer (`robogym.metrics.paper_metrics.score_episode`) implement
the same formulas; both run against the `SimBackend` protocol on real LIBERO
and on the synthetic backend. Per-task workspaces and force thresholds come
from `robogym.augmentation.AugmentationPipeline`, which derives them from the
expert demonstration (95th / 99th force percentiles, EE bounding box).

## Quick start

The core depends only on `numpy`, `scipy`, `matplotlib`, `pandas`, and
`tqdm`:

```bash
pip install -e .
pytest -q                       # 28 pass, 1 skipped (real LIBERO)
python -m robogym.demo          # runs the full demo into ./results
python -m robogym.demo --quick  # fast smoke run
```

The demo exercises the pipeline end-to-end on the synthetic backend with the
expert-replay oracle and the scripted `MockPolicy` surrogates. Output lands
under `results/`:

- `results/raw/`: evaluation CSV tree (`global_summary.csv`,
  `suite_summary.csv`, per-task `task_*/task_details.csv`). The schema
  matches a real-LIBERO run. The numbers come from surrogate policies; they
  are not the paper's reported numbers.
- `results/figures/`: `fig3a_*` and `fig3b_*` (paper Figure 3 radars
  rendered from the transcribed numbers in `paper_results/`), the `diag_*`
  diagnostic plots that visualise Tables 2-5, and `live_*` versions from the
  surrogate-policy run.
- `results/paper_tables/`: Tables 2-6 rendered from the transcribed paper
  numbers as CSV and a combined markdown report.

As a sanity check, the oracle (expert-replay) policy should score around
`SR 1.0`, `Comp ≥ 70`, `Safety ≥ 90` on the synthetic suite. This confirms
the metric engine is wired correctly end-to-end.

## Full evaluation against real models

The `[sim]` extra pulls in robosuite, MuJoCo, h5py, draccus, and imageio:

```bash
pip install -e ".[sim]"
```

The eight models reported in Tables 2-4 are wired into `robogym.policies`:

- `Pi0` / `Pi0.5`: `robogym.policies.OpenPiPolicy` talks to an openpi
  websocket server; the reference runner is
  `robogym/runners/run_libero_eval_pi05.py` (see `scripts/run_pi05.sh`).
- `DM0`: `robogym.policies.model_zoo.DexboticPolicy` against the dexbotic
  playground (`playground/benchmarks/libero/*`).
- `GR00T-N1.6`: `robogym.policies.model_zoo.GR00TPolicy` via the
  Isaac-GR00T inference service.
- `RDT`, `X-VLA`, `GigaBrain0.1`: served behind an HTTP endpoint; point
  `robogym.policies.GenericClientPolicy` at it.
- `LingBot-VLA w/ depth`: `robogym.policies.model_zoo.DepthClientPolicy`
  forwards an additional depth channel.

`robogym.policies.MODEL_REGISTRY` indexes all of these by their paper-table
name. The registry also exposes a legacy `OpenVLA-OFT` adapter (used by
`robogym/runners/run_libero_eval_custom_metrics.py` and
`scripts/run_openvla_oft.sh`); it is provided for convenience and is not
one of the eight evaluated models.

## Repository layout

```
robogym/
  metrics/        TrajectoryEvaluator, dtw, sparc, safety, paper_metrics
  envs/           SimBackend protocol, synthetic backend, physics sim, LIBERO
  policies/       paper-model adapters + scripted mock + oracle
  runners/        unified runner, pi0/pi0.5 runner, legacy OpenVLA runner
  tasks/          50-task System-2 reasoning suite (physics + logic gated)
  augmentation/   Figure-1 pipeline + scoring-script synthesis + repair loop
  data/           §4.1 expert-trajectory generation, HDF5 / RLDS export
  assets/         §3.6 asset generation (text/image-to-3D + textures)
  analysis/       Tables 2-6 / Figure 3 rendering, §4.4 human-study harness
paper_results/    transcribed numbers from the paper (used by the renderer)
tests/            metrics, pipeline, tasks (29 tests)
```
