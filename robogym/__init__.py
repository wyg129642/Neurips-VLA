"""RoboGym: a multi-dimensional evaluation benchmark for VLA models.

The metric is defined by equations (1)-(5) (paper §3.2-3.4) and the
operational-safety term of §3.5; the 50-task System-2 reasoning suite
follows §3.6.

Subpackages:

- :mod:`robogym.metrics` - the streaming multi-dimensional evaluator plus
  the closed-form :mod:`paper_metrics` scorer.
- :mod:`robogym.envs` - simulator backends (a LIBERO wrapper for the full
  stack and a dependency-free synthetic backend used by tests/demo).
- :mod:`robogym.policies` - oracle/mock baselines plus model adapters
  (OpenVLA-OFT, openpi pi0/pi0.5, and generic served VLAs).
- :mod:`robogym.runners` - the unified evaluation runner.
- :mod:`robogym.tasks` - the System-2 reasoning suite.
- :mod:`robogym.analysis` - renders the tables and figures from a
  paper-numbers fixture and from live runner output.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
