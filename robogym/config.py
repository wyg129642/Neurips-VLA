"""Demo-only competence presets for the scripted mock policies.

These are illustrative test fixtures, not paper numbers. Real evaluations
use the adapters in :mod:`robogym.policies.model_zoo`.
"""

from __future__ import annotations

PAPER_MODEL_COMPETENCE = {
    "LingBot-VLA w/ depth": dict(libero=0.93, reasoning=0.40),
    "GigaBrain0.1":         dict(libero=0.92, reasoning=0.46),
    "X-VLA":                dict(libero=0.91, reasoning=0.10),
    "GR00T-N1.6":           dict(libero=0.88, reasoning=0.24),
    "DM0":                  dict(libero=0.89, reasoning=0.50),
    "Pi0.5":                dict(libero=0.86, reasoning=0.30),
    "Pi0":                  dict(libero=0.82, reasoning=0.16),
    "RDT":                  dict(libero=0.80, reasoning=0.12),
}

DEFAULT_DT = 0.05
