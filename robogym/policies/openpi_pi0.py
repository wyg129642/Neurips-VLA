"""pi0 / pi0.5 policy adapter (openpi).

Talks to an openpi policy server over websockets and returns the predicted
action chunk, so pi0 and pi0.5 plug into the unified runner and the
trajectory evaluator.
"""

from __future__ import annotations

import numpy as np

from .base import Policy

class OpenPiPolicy(Policy):
    """Talks to an openpi policy server (``scripts/serve_policy.py``).

    The openpi LIBERO example serves the model over websockets; this client
    mirrors ``examples/libero`` preprocessing (resize to 224, BGR-to-RGB
    handled server-side) and returns the predicted action chunk.
    """

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
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "openpi-client is not installed. Install the openpi-client "
                "package and start a policy server with "
                "`python scripts/serve_policy.py`."
            ) from exc
        self._client = websocket_client_policy.WebsocketClientPolicy(
            host=self.host, port=self.port)

    def reset(self, **kw) -> None:
        self._ensure_client()

    def act(self, observation: dict, task_description: str) -> np.ndarray:
        self._ensure_client()
        # openpi LIBERO element naming (examples/libero/main.py)
        element = {
            "observation/image": observation["full_image"],
            "observation/wrist_image": observation["wrist_image"],
            "observation/state": observation["state"],
            "prompt": task_description,
        }
        result = self._client.infer(element)
        return np.asarray(result["actions"], dtype=float)
