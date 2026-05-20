# RoboGym

Beyond Success Rates in Vision-Language-Action Models via Multi-Dimensional
Evaluation.

RoboGym is a multi-dimensional evaluation benchmark for Vision-Language-Action
(VLA) models that moves beyond a binary success rate. Each rollout is scored on
five dimensions, and the benchmark ships with a 50-task System-2 reasoning
suite that uses test-time randomization of task-critical variables so that
solutions cannot be memorised.

## Evaluation dimensions

| Dimension              | Paper        | Implementation                                              |
|------------------------|--------------|-------------------------------------------------------------|
| Completeness           | §3.2, eq 1/2 | sequence-gated DTW progress against the expert demo         |
| Spatial Efficiency     | §3.3, eq 3   | path-optimality ratio (capped at 1)                         |
| Temporal Efficiency    | §3.3, eq 4   | progress-rate vs. expert speed (capped at 1)                |
| Kinematic Smoothness   | §3.4, eq 5   | Spectral Arc Length (SPARC)                                 |
| Operational Safety     | §3.5         | tripartite workspace / contact / dynamic-force penalty      |

The streaming evaluator (`robogym.metrics.TrajectoryEvaluator`) and the
closed-form `score_episode` function (`robogym.metrics.paper_metrics`) both
implement these formulas, run against the same `SimBackend` protocol on real
LIBERO and on the dependency-free synthetic backend. The §3.1 task-augmentation
pipeline (Figure 1) wraps a binary-success task into a multi-dimensional one by
replaying the expert demo, extracting safe-force / workspace / phase-weight
baselines, and emitting a per-task scoring script. The 50-task System-2
reasoning suite (§3.6, Table 4) spans Symbolic-Geometric, Physical-Intuition,
and Memory-Sequential reasoning, each backed by a physics-grounded simulator.

## Quick start

The core depends only on `numpy`, `scipy`, `matplotlib`, `pandas`, and `tqdm`
(no GPU, model weights, or simulator required):

```bash
pip install -e .
pytest -q                       # 29 tests (28 pass, 1 skipped without real LIBERO)
python -m robogym.demo          # runs the full demo into ./results
python -m robogym.demo --quick  # fast smoke run
```

The demo exercises the pipeline end-to-end on the dependency-free synthetic
backend using the expert-replay oracle and the scripted `MockPolicy` surrogate.
Output is written under `results/`:

- `results/raw/` is the evaluation CSV tree, with `global_summary.csv`,
  `suite_summary.csv`, and per-task `task_*/task_details.csv`. The schema
  matches a real-LIBERO run; the values come from the surrogate policies and
  are not intended to reproduce the paper's reported numbers.
- `results/figures/` contains `fig3a_*` and `fig3b_*` (the paper Figure 3
  radar charts, rendered from the transcribed numbers in `paper_results/`),
  the `diag_*` diagnostic plots that visualise Tables 2-5, and `live_*`
  versions from the surrogate-policy run.
- `results/paper_tables/` contains Tables 2-5 rendered from the transcribed
  paper numbers, as CSV and a combined markdown report.

As a sanity check, the oracle (expert-replay) policy should score around
`SR 1.0`, `Comp 100`, `Smooth 96`, `Safety 100`, `Total 92` on the synthetic
suite. This confirms the metric engine is wired correctly end-to-end.

## Full evaluation against real models

The optional `[sim]` extra pulls in robosuite, MuJoCo, h5py, draccus, and
imageio for the full LIBERO simulation stack:

```bash
pip install -e ".[sim]"
```

Adapters and runners for the eight models evaluated in the paper live under
`robogym.policies`:

- OpenVLA-OFT runs through `robogym/runners/run_libero_eval_custom_metrics.py`;
  see `scripts/run_openvla_oft.sh` for an example invocation.
- pi0 and pi0.5 use `robogym.policies.OpenPiPolicy` against an openpi
  websocket server. The reference runner is
  `robogym/runners/run_libero_eval_pi05.py` (see `scripts/run_pi05.sh`).
- DM0, CogAct, and MemVLA are served via the dexbotic playground; the adapter
  is `robogym.policies.model_zoo.DexboticPolicy`.
- GR00T-N1.6 uses the Isaac-GR00T inference service via
  `robogym.policies.model_zoo.GR00TPolicy`.
- RDT, X-VLA, and GigaBrain0.1 are served behind HTTP endpoints; point
  `robogym.policies.GenericClientPolicy` at the endpoint.
- LingBot-VLA-w/-depth uses
  `robogym.policies.model_zoo.DepthClientPolicy`, which adds a depth channel
  to the same protocol.

`robogym.policies.MODEL_REGISTRY` collects all of these by paper-table name.
The multi-dimensional evaluator runs unchanged on real LIBERO via
`robogym.envs.LiberoBackend` (the simulator is accessed through the backend
abstraction).

## Repository layout

```
robogym/
  metrics/        TrajectoryEvaluator, dtw, sparc, safety, paper_metrics
  envs/           SimBackend protocol, synthetic backend, physics sim, real LIBERO
  policies/       oracle, mock, OpenVLA-OFT, openpi pi0/pi0.5, model registry
  runners/        unified runner, OpenVLA-OFT runner, pi0.5 runner
  tasks/          50-task System-2 reasoning suite (physics- and logic-gated)
  augmentation/   §3.1 pipeline plus the code-agent (synthesise + repair loop)
  data/           §4.1 expert-trajectory generation, HDF5 and RLDS exports
  assets/         §3.6 asset generation (text/image-to-3D plus textures)
  analysis/       Tables 2-5 / Figure 3 rendering and §4.4 human-study harness
paper_results/    transcribed numbers from the paper (used by the renderer)
tests/            metrics, pipeline, tasks (29 tests)
```
