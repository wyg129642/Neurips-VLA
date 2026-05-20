"""Real adapters and registry for the eight VLA models in the paper.

The paper compares OpenVLA-OFT, pi0, pi0.5, GR00T-N1.6, RDT, X-VLA, DM0,
GigaBrain0.1, and LingBot-VLA-w/-depth. Each adapter wraps the inference
API of that model family (documented per class) and is import-guarded; the
registry lets the unified runner evaluate any of them once the
corresponding server or checkpoint is available. For the dependency-free
demo the :class:`MockPolicy` fixtures stand in; see :mod:`robogym.config`
for the per-model competence constants those surrogates use.

Serving patterns (what each adapter needs on a full-stack machine):

- pi0 / pi0.5: openpi websocket server (``scripts/serve_policy.py``).
- DM0 / CogAct / MemVLA: dexbotic playground inference (the
  ``playground/benchmarks/libero/*`` configs in the dexbotic repo).
- GR00T-N1.6: NVIDIA Isaac-GR00T inference service.
- RDT / X-VLA / GigaBrain0.1 / LingBot-VLA: HTTP inference endpoint
  (LingBot additionally consumes a depth channel).
"""

from __future__ import annotations

import numpy as np

from .base import Policy
from .generic_client import GenericClientPolicy
from .openpi_pi0 import OpenPiPolicy

class DexboticPolicy(Policy):
    """DM0 / CogAct / MemVLA served through the dexbotic playground.

    Mirrors the dexbotic ``playground/benchmarks/libero`` inference config
    (``--task inference``). Requires the dexbotic package and the model
    checkpoint.
    """

    def __init__(self, exp: str = "libero_pi0", checkpoint: str = "",
                 model_name: str = "DM0"):
        self.exp, self.checkpoint, self.name = exp, checkpoint, model_name
        self._model = None

    def _ensure(self):
        if self._model is not None:
            return
        try:
            from dexbotic.api import InferenceRunner  # type: ignore
        except ImportError as exc:  # pragma: no cover - full stack only
            raise RuntimeError(
                "dexbotic is not installed. Run the dexbotic playground "
                f"benchmark '{self.exp}' with --task inference, or serve it "
                "behind an endpoint and use GenericClientPolicy instead."
            ) from exc
        self._model = InferenceRunner(self.exp, self.checkpoint)

    def reset(self, **kw):
        self._ensure()

    def act(self, observation, task_description):
        self._ensure()
        return np.asarray(self._model.infer(
            observation, task_description), float)

class GR00TPolicy(GenericClientPolicy):
    """GR00T-N1.6 via the Isaac-GR00T inference service (HTTP)."""

    def __init__(self, endpoint: str = "http://0.0.0.0:5555/act"):
        super().__init__(endpoint, model_name="GR00T-N1.6")

class DepthClientPolicy(GenericClientPolicy):
    """LingBot-VLA-w/-depth: like the generic client but also forwards a
    depth image when the backend provides one."""

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

# name -> factory(**kwargs) -> Policy. Mirrors the paper's model table.
MODEL_REGISTRY = {
    "OpenVLA-OFT": lambda **k: _lazy_openvla(**k),
    "Pi0": lambda **k: OpenPiPolicy(model_name="Pi0", **k),
    "Pi0.5": lambda **k: OpenPiPolicy(model_name="Pi0.5", **k),
    "DM0": lambda **k: DexboticPolicy(model_name="DM0",
                                      exp="libero_pi0", **k),
    "CogAct": lambda **k: DexboticPolicy(model_name="CogAct",
                                         exp="libero_cogact", **k),
    "MemVLA": lambda **k: DexboticPolicy(model_name="MemVLA",
                                         exp="libero_memvla", **k),
    "GR00T-N1.6": lambda **k: GR00TPolicy(**k),
    "RDT": lambda **k: GenericClientPolicy(model_name="RDT", **k),
    "X-VLA": lambda **k: GenericClientPolicy(model_name="X-VLA", **k),
    "GigaBrain0.1": lambda **k: GenericClientPolicy(
        model_name="GigaBrain0.1", **k),
    "LingBot-VLA w/ depth": lambda **k: DepthClientPolicy(**k),
}

def _lazy_openvla(**kw):
    from .openvla_oft import OpenVLAOFTPolicy
    return OpenVLAOFTPolicy(**kw)

def make_policy(model_name: str, **kwargs) -> Policy:
    """Construct the real adapter for ``model_name`` (paper model table)."""
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {model_name!r}; "
                       f"known: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_name](**kwargs)

def make_mock_fleet(seed: int = 0):
    """Demo/test stand-ins: the 8 model names bound to calibrated
    :class:`MockPolicy` surrogates (ranking tracks Tables 2-3)."""
    from ..config import PAPER_MODEL_COMPETENCE
    from .mock import MockPolicy

    fleet = {}
    for name, comp in PAPER_MODEL_COMPETENCE.items():
        fleet[name] = MockPolicy(competence=comp["libero"],
                                 seed=seed + hash(name) % 9999,
                                 drop_prob=0.1, name=name)
    return fleet
