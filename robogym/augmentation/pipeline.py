"""Task-augmentation pipeline (paper Figure 1, §3.1).

Implements the Figure 1 loop: take a binary-success task (LIBERO or
otherwise), replay its expert demo to build the execution log
(end-effector / object / contact data), synthesise the multi-dimensional
scoring scripts via :mod:`scoring_synthesizer`, and attach them to the env
as the augmentation record. The result is a
:class:`robogym.envs.base.RoboGymEnv` that the unified runner and streaming
evaluator can score on all five dimensions instead of a 0/1 success bit.
"""

from __future__ import annotations

import numpy as np

from ..envs.base import RoboGymEnv
from ..metrics.trajectory_evaluator import TrajectoryEvaluator
from .scoring_synthesizer import OfflineTemplateSynthesizer

class AugmentationPipeline:
    """Augment any task into a multi-dimensional :class:`RoboGymEnv`."""

    def __init__(self, synthesizer=None, synthesize_code: bool = False,
                 code_agent=None):
        self.synthesizer = synthesizer or OfflineTemplateSynthesizer()
        self.synthesize_code = synthesize_code
        self._code_agent = code_agent

    def augment(self, backend, expert_actions: np.ndarray,
                expert_init_state: np.ndarray, task_description: str,
                dt: float = 0.05) -> RoboGymEnv:
        """One pass of the Figure-1 pipeline for a single task."""
        # 1. Execution Log + reference paths (reuse the extractor so
        #    the augmentation record is consistent with how it'll be scored).
        ev = TrajectoryEvaluator(backend, expert_actions, expert_init_state,
                                 dt=dt)
        # 2. AI-Driven Script Synthesis -> Scoring Scripts.
        record = self.synthesizer.synthesize(
            task_description=task_description,
            expert_ee_path=ev.expert_ee_path,
            tracked_objects=ev.tracked_objects,
            expert_forces=None)
        record["reach_weight"] = float(ev.reach_weight)
        record["tracked_objects"] = [o["name"] for o in ev.tracked_objects]
        # 2b. Code Repair loop: synthesize+execute+repair a real scoring
        #     script (the Fig-1 "AI-Driven Script Synthesis" + "Code Repair").
        if self.synthesize_code:
            from .code_agent import CodeRepairLoop

            agent = self._code_agent or CodeRepairLoop()
            res = agent.run(record)
            record["scoring_script"] = {
                "path": res.path, "valid": res.valid,
                "repair_rounds": res.rounds, "repaired": res.repaired,
            }
        # 3. Attach scoring metadata to the env (the Fig-1 output arrow).
        return RoboGymEnv(backend, task_description=task_description,
                          augmentation=record, dt=dt)

    def augment_suite(self, suite, num_probe_demos: int = 1) -> dict[int, dict]:
        """Augment every task in a runner suite; return ``{task_id: record}``."""
        records = {}
        for task_id in range(suite.n_tasks):
            env = suite.make_env(task_id)
            ea, eis = suite.expert_demo(env, 0)
            aug = self.augment(env.backend, ea, eis,
                               suite.task_spec(task_id).task_description,
                               dt=getattr(suite, "dt", 0.05))
            records[task_id] = aug.augmentation
        return records
