"""Adapters and registry for the VLA models evaluated in the paper.

The eight models in Tables 2-4 are: LingBot-VLA w/ depth, GigaBrain0.1,
X-VLA, GR00T-N1.6, DM0, Pi0.5, Pi0, and RDT. Each adapter wraps the
inference API of one model family; ``MODEL_REGISTRY`` collects them by
the name the paper uses.
"""

from __future__ import annotations

import numpy as np

from .base import Policy
from .generic_client import GenericClientPolicy
from .openpi_pi0 import OpenPiPolicy


class DexboticPolicy(Policy):
    """DM0 served through the dexbotic playground."""

    def __init__(self, exp: str = "libero_dm0", checkpoint: str = "",
                 model_name: str = "DM0"):
        self.exp, self.checkpoint, self.name = exp, checkpoint, model_name
        self._model = None

    def _ensure(self):
        if self._model is not None:
            return
        try:
            from dexbotic.api import InferenceRunner
        except ImportError as exc:
            raise RuntimeError(
                "dexbotic is not installed; serve the model behind an HTTP "
                "endpoint and use GenericClientPolicy instead."
            ) from exc
        self._model = InferenceRunner(self.exp, self.checkpoint)

    def reset(self, **kw):
        self._ensure()

    def act(self, observation, task_description):
        self._ensure()
        return np.asarray(self._model.infer(
            observation, task_description), float)


class GR00TPolicy(GenericClientPolicy):
    """GR00T-N1.6 via the Isaac-GR00T inference service."""

    def __init__(self, endpoint: str = "http://0.0.0.0:5555/act"):
        super().__init__(endpoint, model_name="GR00T-N1.6")


class DepthClientPolicy(GenericClientPolicy):
    """LingBot-VLA-w/-depth: forwards an extra depth channel."""

    def __init__(self, endpoint: str = "http://0.0.0.0:6000/act", **kw):
        kw.pop("model_name", None)
        super().__init__(endpoint, model_name="LingBot-VLA w/ depth", **kw)

    def act(self, observation, task_description):
        import requests

        payload = {"task_description": task_description,
                   "state": np.asarray(observation["state"]).tolist()}
        if "full_image" in observation:
            from .generic_client import _encode_image
            payload["full_image"] = _encode_image(observation["full_image"])
            if observation.get("depth_image") is not None:
                payload["depth_image"] = _encode_image(
                    observation["depth_image"])
        r = requests.post(self.endpoint, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return np.asarray(r.json()["actions"], float)


def _lazy_openvla(**kw):
    from .openvla_oft import OpenVLAOFTPolicy
    return OpenVLAOFTPolicy(**kw)


# Adapters for the eight models reported in Tables 2-4. The OpenVLA-OFT
# adapter is retained as a legacy convenience for users who still run that
# baseline; it is not one of the eight evaluated models.
MODEL_REGISTRY = {
    "Pi0":            lambda **k: OpenPiPolicy(model_name="Pi0", **k),
    "Pi0.5":          lambda **k: OpenPiPolicy(model_name="Pi0.5", **k),
    "DM0":            lambda **k: DexboticPolicy(model_name="DM0",
                                                 exp="libero_dm0", **k),
    "GR00T-N1.6":     lambda **k: GR00TPolicy(**k),
    "RDT":            lambda **k: GenericClientPolicy(model_name="RDT", **k),
    "X-VLA":          lambda **k: GenericClientPolicy(model_name="X-VLA", **k),
    "GigaBrain0.1":   lambda **k: GenericClientPolicy(
        model_name="GigaBrain0.1", **k),
    "LingBot-VLA w/ depth": lambda **k: DepthClientPolicy(**k),
    # Legacy (not in the paper).
    "OpenVLA-OFT":    lambda **k: _lazy_openvla(**k),
}


def make_policy(model_name: str, **kwargs) -> Policy:
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {model_name!r}; "
                       f"known: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_name](**kwargs)


def make_mock_fleet(seed: int = 0):
    """Scripted MockPolicy stand-ins keyed by the paper's eight model names."""
    from ..config import PAPER_MODEL_COMPETENCE
    from .mock import MockPolicy

    fleet = {}
    for name, comp in PAPER_MODEL_COMPETENCE.items():
        fleet[name] = MockPolicy(competence=comp["libero"],
                                 seed=seed + hash(name) % 9999,
                                 drop_prob=0.1, name=name)
    return fleet
