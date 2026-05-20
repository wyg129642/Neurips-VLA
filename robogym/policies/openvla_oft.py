"""OpenVLA-OFT policy adapter.

Wraps the model-loading and ``get_action`` path used by the OpenVLA-OFT
runner (the ``experiments.robot.*`` helpers from the openvla-oft repo).
Imports are deferred so this module stays importable on machines that do
not have torch, transformers, or the openvla weights.
"""

from __future__ import annotations

import numpy as np

from .base import Policy

class OpenVLAOFTPolicy(Policy):
    """OpenVLA-OFT adapter.

    ``cfg`` is the ``GenerateConfig`` (or a duck-typed equivalent).
    Model components are loaded lazily on first :meth:`reset`.
    """

    name = "openvla_oft"

    def __init__(self, cfg):
        self.cfg = cfg
        self._loaded = False
        self._model = self._processor = self._action_head = None
        self._proprio = self._noisy = None
        self._resize = None

    def _ensure_loaded(self):
        if self._loaded:
            return
        # Mirrors initialize_model() in the OpenVLA-OFT runner.
        from experiments.robot.robot_utils import (
            get_image_resize_size, get_model)
        from experiments.robot.openvla_utils import (
            get_action_head, get_noisy_action_projector, get_processor,
            get_proprio_projector)

        self._model = get_model(self.cfg)
        self._proprio = (get_proprio_projector(self.cfg, self._model.llm_dim,
                                               proprio_dim=8)
                         if self.cfg.use_proprio else None)
        self._action_head = (get_action_head(self.cfg, self._model.llm_dim)
                             if (self.cfg.use_l1_regression
                                 or self.cfg.use_diffusion) else None)
        self._noisy = (get_noisy_action_projector(self.cfg,
                                                  self._model.llm_dim)
                       if self.cfg.use_diffusion else None)
        self._processor = (get_processor(self.cfg)
                           if self.cfg.model_family == "openvla" else None)
        self._resize = get_image_resize_size(self.cfg)
        self._loaded = True

    def reset(self, **kw) -> None:
        self._ensure_loaded()

    def act(self, observation: dict, task_description: str) -> np.ndarray:
        from experiments.robot.robot_utils import get_action
        actions = get_action(
            self.cfg, self._model, observation, task_description,
            processor=self._processor, action_head=self._action_head,
            proprio_projector=self._proprio,
            noisy_action_projector=self._noisy,
            use_film=self.cfg.use_film)
        return np.asarray(actions, dtype=float)

    @property
    def resize_size(self):
        self._ensure_loaded()
        return self._resize
