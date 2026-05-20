"""RoboGym task-augmentation pipeline (paper Figure 1 / §3.1).

Wraps any binary-success task into a multi-dimensional
:class:`~robogym.envs.base.RoboGymEnv` by replaying the expert demo and
synthesizing a per-task scoring script. The default synthesizer is a
deterministic offline template; an optional LLM-driven synthesizer is
available for the Gemini Code Agent variant described in the paper.
"""

from .code_agent import CodeRepairLoop, RepairResult, synthesize
from .pipeline import AugmentationPipeline
from .scoring_synthesizer import (
    LLMScoringSynthesizer,
    OfflineTemplateSynthesizer,
)

__all__ = [
    "AugmentationPipeline",
    "OfflineTemplateSynthesizer",
    "LLMScoringSynthesizer",
    "CodeRepairLoop",
    "RepairResult",
    "synthesize",
]
