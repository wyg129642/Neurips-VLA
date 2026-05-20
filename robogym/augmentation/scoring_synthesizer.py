"""Scoring-script / augmentation-record synthesis (paper Figure 1, §3.1).

The augmentation record (workspace, force thresholds, phase weights, ...)
parameterises the per-task scoring script used by the metric engine.

Two backends:

- :class:`OfflineTemplateSynthesizer` (default): derives the record
  deterministically from expert-demo statistics, with no network or LLM
  call. This is what makes the pipeline reproducible and runnable without
  the heavy simulation stack.
- :class:`LLMScoringSynthesizer`: optional. Sends the task description and
  expert baselines to an LLM (Gemini by default, matching the paper's
  "AI-Driven Script Synthesis" stage) and lets it override a small set of
  scalar fields. Falls back to the offline template if no API key is set.
"""

from __future__ import annotations

import numpy as np

def _expert_stats(expert_ee_path: np.ndarray, expert_forces: np.ndarray | None,
                   tracked_objects: list[dict]) -> dict:
    """Critical baselines extracted from the expert demo (Fig 1 'Execution
    Log' -> 'Multi-dimensional Scoring Scripts')."""
    ee = np.asarray(expert_ee_path, float)
    lo = ee.min(axis=0) - 0.15            # safe workspace W (§3.5) with margin
    hi = ee.max(axis=0) + 0.15
    seg = np.linalg.norm(np.diff(ee, axis=0), axis=1) if len(ee) > 1 \
        else np.array([0.0])
    expert_len = float(seg.sum())
    if expert_forces is not None and len(expert_forces):
        f = np.asarray(expert_forces, float)
        f_sus = float(np.percentile(f, 95))           # §3.5 sustained-force
        f_peak = float(np.percentile(f, 99))          # §3.5 peak-force
    else:
        f_sus, f_peak = 40.0, 60.0                     # defaults
    # phase weights proportional to per-phase expert displacement (eq 2 w_i, sum=1)
    disps = [o["total_disp"] for o in tracked_objects]
    reach = max(expert_len - sum(disps), 1e-6)
    raw = np.array([reach] + disps, float)
    weights = (raw / raw.sum()).tolist()
    return {
        "workspace": [lo.tolist(), hi.tolist()],
        "expert_total_len": expert_len,
        "f_sus_lim": f_sus, "f_peak_lim": f_peak,
        "phase_weights": weights,
        "num_phases": len(raw),
        "gamma_expert": 1.0,
    }

class OfflineTemplateSynthesizer:
    """Deterministic augmentation-record generator (no LLM)."""

    name = "offline_template"

    def synthesize(self, task_description: str, expert_ee_path: np.ndarray,
                    tracked_objects: list[dict],
                    expert_forces: np.ndarray | None = None) -> dict:
        rec = _expert_stats(expert_ee_path, expert_forces, tracked_objects)
        rec["task_description"] = task_description
        rec["synthesizer"] = self.name
        return rec

class LLMScoringSynthesizer:
    """LLM-driven synthesizer for the paper's AI-Driven Script Synthesis stage.

    Sends the task semantics and expert statistics to an LLM and asks for
    overridden metric weights and safety thresholds as JSON. Any failure
    (missing key, network error, parse error) falls back transparently to
    the offline template so the pipeline is never blocked.
    """

    name = "llm_gemini"

    def __init__(self, model: str = "gemini-2.0", api_key: str | None = None):
        import os
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") \
            or os.environ.get("GOOGLE_API_KEY")
        self._fallback = OfflineTemplateSynthesizer()

    def synthesize(self, task_description: str, expert_ee_path: np.ndarray,
                   tracked_objects: list[dict],
                   expert_forces: np.ndarray | None = None) -> dict:
        base = self._fallback.synthesize(task_description, expert_ee_path,
                                         tracked_objects, expert_forces)
        if not self.api_key:
            base["synthesizer"] = f"{self.name}:fallback(offline)"
            return base
        try:  # pragma: no cover - needs network + key
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            prompt = (
                "You are RoboGym's scoring-script synthesizer. Given the task "
                f"'{task_description}' and these expert baselines {base!r}, "
                "return a JSON object overriding any of: phase_weights (sum=1), "
                "f_sus_lim, f_peak_lim, gamma_expert. JSON only.")
            import json
            txt = genai.GenerativeModel(self.model).generate_content(
                prompt).text
            override = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
            base.update({k: v for k, v in override.items() if k in base})
            base["synthesizer"] = self.name
        except Exception as exc:
            base["synthesizer"] = f"{self.name}:fallback({type(exc).__name__})"
        return base
