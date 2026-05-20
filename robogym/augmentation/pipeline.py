"""Figure-1 task-augmentation pipeline."""

from __future__ import annotations

import numpy as np

from ..envs.base import RoboGymEnv
from ..metrics.trajectory_evaluator import TrajectoryEvaluator
from .scoring_synthesizer import OfflineTemplateSynthesizer


class AugmentationPipeline:
    """Turn a binary-success task into a multi-dimensional :class:`RoboGymEnv`."""

    def __init__(self, synthesizer=None, synthesize_code: bool = False,
                 code_agent=None):
        self.synthesizer = synthesizer or OfflineTemplateSynthesizer()
        self.synthesize_code = synthesize_code
        self._code_agent = code_agent

    def augment(self, backend, expert_actions: np.ndarray,
                expert_init_state: np.ndarray, task_description: str,
                dt: float = 0.05) -> RoboGymEnv:
        ev = TrajectoryEvaluator(backend, expert_actions, expert_init_state,
                                 dt=dt)
        record = self.synthesizer.synthesize(
            task_description=task_description,
            expert_ee_path=ev.expert_ee_path,
            tracked_objects=ev.tracked_objects,
            expert_forces=None)
        record["reach_weight"] = float(ev.reach_weight)
        record["tracked_objects"] = [o["name"] for o in ev.tracked_objects]
        if self.synthesize_code:
            from .code_agent import CodeRepairLoop

            agent = self._code_agent or CodeRepairLoop()
            res = agent.run(record)
            record["scoring_script"] = {
                "path": res.path, "valid": res.valid,
                "repair_rounds": res.rounds, "repaired": res.repaired,
            }
        return RoboGymEnv(backend, task_description=task_description,
                          augmentation=record, dt=dt)

    def augment_suite(self, suite, num_probe_demos: int = 1) -> dict[int, dict]:
        records = {}
        for task_id in range(suite.n_tasks):
            env = suite.make_env(task_id)
            ea, eis = suite.expert_demo(env, 0)
            aug = self.augment(env.backend, ea, eis,
                               suite.task_spec(task_id).task_description,
                               dt=getattr(suite, "dt", 0.05))
            records[task_id] = aug.augmentation
        return records
