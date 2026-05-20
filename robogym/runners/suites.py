"""Task providers for the unified runner.

Exposes a "give me task ``k``'s env plus ``N`` expert demos" abstraction so
one runner can drive both the dependency-free synthetic backend and real
LIBERO (via the ``ExpertDataLoader`` HDF5 logic). Synthetic task names
mirror the ``libero_object`` suite exactly as seen in the result CSVs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from ..envs.base import RoboGymEnv
from ..envs.synthetic_env import SyntheticPickPlaceSim

# Object names taken from the suite_summary.csv so the
# synthetic suite reproduces the task list 1:1.
LIBERO_OBJECT_TASKS = [
    "alphabet_soup", "cream_cheese", "salad_dressing", "bbq_sauce", "ketchup",
    "tomato_sauce", "butter", "milk", "chocolate_pudding", "orange_juice",
]

# Augmented-LIBERO suite sizes (paper §4.1 / §4.2 "Augmented LIBERO"). On a
# full-stack machine the task names come from the ``libero`` benchmark dict
# via :class:`LiberoSuite`; the synthetic backend here matches the suite
# task counts as a clearly-labelled stand-in so the augmentation and
# multi-dimensional pipeline can be exercised on every suite.
LIBERO_SUITE_SIZES = {
    "libero_spatial": 10, "libero_object": 10, "libero_goal": 10,
    "libero_10": 10, "libero_90": 90,
}

def _synthetic_objects(suite_name: str) -> list[str]:
    if suite_name == "libero_object":
        return list(LIBERO_OBJECT_TASKS)
    n = LIBERO_SUITE_SIZES.get(suite_name, 10)
    return [f"{suite_name}_item_{i:02d}" for i in range(n)]

@dataclass
class TaskSpec:
    task_id: int
    task_name: str
    task_description: str

class SyntheticSuite:
    """A synthetic stand-in for a LIBERO suite (default: ``libero_object``)."""

    def __init__(self, suite_name: str = "libero_object",
                 objects: list[str] | None = None, dt: float = 0.05,
                 seed: int = 0):
        self.suite_name = suite_name
        self.objects = objects or _synthetic_objects(suite_name)
        self.dt = dt
        self.seed = seed

    @property
    def n_tasks(self) -> int:
        return len(self.objects)

    def task_spec(self, task_id: int) -> TaskSpec:
        obj = self.objects[task_id]
        name = f"pick_up_the_{obj}_and_place_it_in_the_basket"
        desc = f"pick up the {obj.replace('_', ' ')} and place it in the basket"
        return TaskSpec(task_id, name, desc)

    def make_env(self, task_id: int) -> RoboGymEnv:
        obj = self.objects[task_id]
        sim = SyntheticPickPlaceSim(task_objects=[obj], target_name="basket",
                                    seed=self.seed + task_id, dt=self.dt)
        return RoboGymEnv(sim, task_description=self.task_spec(
            task_id).task_description, dt=self.dt)

    def expert_demo(self, env: RoboGymEnv, demo_idx: int,
                    jitter: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        return env.backend.generate_expert_demo(jitter=jitter,
                                                seed=self.seed * 1000
                                                + demo_idx)

class LiberoSuite:
    """Real-LIBERO provider reusing the ``ExpertDataLoader`` HDF5
    schema (``data/demo_i/{actions,states}``)."""

    def __init__(self, suite_name: str, datasets_root: str, dt: float = 0.05,
                 model_family: str = "openvla", resolution: int = 256):
        self.suite_name = suite_name
        self.datasets_root = datasets_root
        self.dt = dt
        self.model_family = model_family
        self.resolution = resolution
        from libero.libero import benchmark  # heavy; deferred
        self._suite = benchmark.get_benchmark_dict()[suite_name]()
        self._names = self._suite.get_task_names()

    @property
    def n_tasks(self) -> int:
        return self._suite.n_tasks

    def task_spec(self, task_id: int) -> TaskSpec:
        task = self._suite.get_task(task_id)
        return TaskSpec(task_id, self._names[task_id], task.language)

    def make_env(self, task_id: int) -> RoboGymEnv:
        from ..envs.libero_env import LiberoBackend, make_libero_env
        env, desc = make_libero_env(self._suite.get_task(task_id),
                                    self.model_family, self.resolution)
        be = LiberoBackend(env)
        rg = RoboGymEnv(be, task_description=desc, dt=self.dt)
        rg._task_id = task_id  # remembered for expert HDF5 lookup
        return rg

    def expert_demo(self, env: RoboGymEnv, demo_idx: int,
                    jitter: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Load ``(actions, init_state)`` of demo ``demo_idx`` from the
        HDF5 convention ``{root}/{suite}/{task_name}_demo.hdf5`` with
        keys ``data/demo_i/{actions,states}`` (== ``ExpertDataLoader``)."""
        import h5py

        task_name = self._names[getattr(env, "_task_id", 0)]
        path = get_expert_hdf5(self.datasets_root, self.suite_name, task_name)
        with h5py.File(path, "r") as f:
            keys = sorted(f["data"].keys(),
                          key=lambda x: int(x.split("_")[1]))
            k = keys[min(demo_idx, len(keys) - 1)]
            actions = f["data"][k]["actions"][:]
            init_state = f["data"][k]["states"][0]
        return np.asarray(actions, float), np.asarray(init_state, float)

def get_expert_hdf5(datasets_root: str, suite_name: str, task_name: str):
    """``ExpertDataLoader`` path convention (full-stack)."""
    return os.path.join(datasets_root, suite_name, f"{task_name}_demo.hdf5")

class ReasoningRunnerSuite:
    """Adapt the 50-task System-2 reasoning suite to the runner's provider API.

    Each task supplies its own pure-numpy sim + algorithmic-oracle expert demo,
    so the evaluator scores reasoning tasks exactly as it scores
    augmented LIBERO (paper §3.6 / Table 3-4).
    """

    def __init__(self, suite_name: str = "system2_reasoning",
                 dt: float = 0.05, seed: int = 0,
                 category: str | None = None):
        from ..tasks.registry import (
            build_reasoning_suite,
            reasoning_suite_by_category,
        )
        self.suite_name = suite_name
        self.dt = dt
        self.seed = seed
        if category:
            self._tasks = reasoning_suite_by_category(seed)[category]
        else:
            self._tasks = build_reasoning_suite(seed)

    @property
    def n_tasks(self) -> int:
        return len(self._tasks)

    def task_spec(self, task_id: int) -> TaskSpec:
        t = self._tasks[task_id]
        return TaskSpec(task_id, t.name, t.description)

    def make_env(self, task_id: int) -> RoboGymEnv:
        env = self._tasks[task_id].make_env()
        env._rg_task_id = task_id          # remember which task this env is
        return env

    def expert_demo(self, env: RoboGymEnv, demo_idx: int,
                    jitter: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Per-demo test-time randomization (§3.6).

        Re-randomises the task and rebinds ``env.backend`` to the freshly
        randomised sim, so the trajectory evaluator's expert pre-roll and
        the policy rollout run on the same instance the expert demo was
        generated from.
        """
        idx = getattr(env, "_rg_task_id", 0)
        t = self._tasks[idx]
        seed = self.seed * 7919 + t.task_id * 101 + demo_idx
        t.randomize(seed)
        sim = t.make_sim()
        env.backend = sim                  # critical: keep env <-> demo in sync
        env.task_description = t.description
        return sim.generate_expert_demo(seed=seed, jitter=jitter)
