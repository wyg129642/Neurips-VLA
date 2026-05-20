"""Task-augmentation pipeline (Sec. 3.1, Figure 1)."""

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
