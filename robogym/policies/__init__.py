"""Policy adapters: oracle and mock baselines plus the paper's eight models."""

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
