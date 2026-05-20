"""pi0 / pi0.5 adapter (openpi websocket client)."""

from __future__ import annotations

import numpy as np

from .base import Policy


class OpenPiPolicy(Policy):
    def __init__(self, host: str = "0.0.0.0", port: int = 8000,
                 model_name: str = "pi05"):
        self.host, self.port = host, port
        self.name = model_name
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            from openpi_client import websocket_client_policy
        except ImportError as exc:
            raise RuntimeError(
                "openpi-client is not installed. Install it and start a "
                "policy server with `python scripts/serve_policy.py`."
            ) from exc
        self._client = websocket_client_policy.WebsocketClientPolicy(
            host=self.host, port=self.port)

    def reset(self, **kw) -> None:
        self._ensure_client()

    def act(self, observation: dict, task_description: str) -> np.ndarray:
        self._ensure_client()
        element = {
            "observation/image": observation["full_image"],
            "observation/wrist_image": observation["wrist_image"],
            "observation/state": observation["state"],
            "prompt": task_description,
        }
        result = self._client.infer(element)
        return np.asarray(result["actions"], dtype=float)
