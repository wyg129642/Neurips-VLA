"""Central configuration knobs and demo/test policy fixtures.

The unified runner evaluates real VLA models through the adapters in
:mod:`robogym.policies`. The dependency-free demo and the test suite run
without GPUs, model weights, or the heavy simulation stack, so the eight
model names below are bound to scripted
:class:`~robogym.policies.MockPolicy` fixtures with preset competence
levels.

These are illustrative test fixtures only. They are not model evaluations
and do not reproduce the paper's reported numbers; the reported numbers
come from real runs of each model on real LIBERO.
"""

from __future__ import annotations

# Preset competence in [0,1] for the scripted MockPolicy demo/test fixtures.
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

DEFAULT_DT = 0.05  # LIBERO control period (20 Hz)
