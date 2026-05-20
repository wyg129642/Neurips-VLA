"""RoboGym policy adapters.

Always available (no heavy deps):

- :class:`OraclePolicy`: replays the expert demonstration (sanity baseline).
- :class:`MockPolicy`: a tunable degraded surrogate used by the demo and tests.
- :class:`GenericClientPolicy`: any served model over HTTP/JSON.

Lazy adapters (require the model's runtime stack):

- :class:`OpenVLAOFTPolicy`: OpenVLA-OFT.
- :class:`OpenPiPolicy`: openpi pi0 / pi0.5.
- :func:`make_policy` and ``MODEL_REGISTRY``: adapters for the eight
  models evaluated in the paper (pi0/pi0.5, DM0/CogAct/MemVLA via dexbotic,
  GR00T, RDT, X-VLA, GigaBrain, LingBot-VLA). :func:`make_mock_fleet`
  returns a parallel fleet of `MockPolicy` surrogates for the demo.
"""

from .base import Policy
from .generic_client import GenericClientPolicy
from .mock import MockPolicy
from .oracle import OraclePolicy

__all__ = [
    "Policy", "OraclePolicy", "MockPolicy", "GenericClientPolicy",
    "make_policy", "make_mock_fleet", "MODEL_REGISTRY",
]

def __getattr__(name):
    if name == "OpenVLAOFTPolicy":
        from .openvla_oft import OpenVLAOFTPolicy
        return OpenVLAOFTPolicy
    if name == "OpenPiPolicy":
        from .openpi_pi0 import OpenPiPolicy
        return OpenPiPolicy
    if name in ("make_policy", "make_mock_fleet", "MODEL_REGISTRY"):
        from . import model_zoo
        return getattr(model_zoo, name)
    raise AttributeError(name)
