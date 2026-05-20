"""Generic HTTP/JSON policy-server client.

Most of the models the paper evaluates (DM0, GigaBrain0.1, GR00T-N1.6,
RDT, X-VLA, LingBot-VLA, and CogAct / MemVLA via dexbotic) are served
behind an inference endpoint rather than imported in-process. This
adapter posts the observation to a user-provided endpoint and expects an
action chunk in response, so any served model plugs into the unified
runner and trajectory evaluator.
"""

from __future__ import annotations

import base64
import io

import numpy as np

from .base import Policy

def _encode_image(img: np.ndarray) -> str:
    """PNG-encode and base64-wrap an image so arbitrary servers can decode."""
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - falls back when Pillow is missing
        return base64.b64encode(np.asarray(img, np.uint8).tobytes()).decode()
    buf = io.BytesIO()
    Image.fromarray(np.asarray(img, np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

class GenericClientPolicy(Policy):
    def __init__(self, endpoint: str, model_name: str = "served_vla",
                 timeout: float = 30.0, send_images: bool = True):
        self.endpoint = endpoint
        self.name = model_name
        self.timeout = timeout
        self.send_images = send_images

    def act(self, observation: dict, task_description: str) -> np.ndarray:
        import requests  # deferred; only needed when actually serving

        payload = {
            "task_description": task_description,
            "state": np.asarray(observation["state"]).tolist(),
        }
        if self.send_images and "full_image" in observation:
            payload["full_image"] = _encode_image(observation["full_image"])
            if "wrist_image" in observation:
                payload["wrist_image"] = _encode_image(
                    observation["wrist_image"])
        r = requests.post(self.endpoint, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return np.asarray(r.json()["actions"], dtype=float)
