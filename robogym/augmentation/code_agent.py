"""Synthesize-execute-repair loop for the per-task scoring script."""

from __future__ import annotations

import importlib.util
import json
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_REQUIRED_KEYS = {"total_fwdbias", "completion", "space_eff", "time_eff",
                  "smoothness", "safety"}


def _baseline_scorer_source(record: dict) -> str:
    rec = json.dumps({
        "phase_weights": record.get("phase_weights", [1.0]),
        "expert_total_len": record.get("expert_total_len", 1.0),
        "f_sus_lim": record.get("f_sus_lim", 40.0),
        "f_peak_lim": record.get("f_peak_lim", 60.0),
        "gamma_expert": record.get("gamma_expert", 1.0),
    })
    return f'''"""Per-task scoring module emitted by code_agent.

Task: {record.get("task_description", "")!r}
"""
import numpy as np
from robogym.metrics.paper_metrics import score_episode

RECORD = {rec}

def compute_scores(episode: dict) -> dict:
    return score_episode(
        np.asarray(episode["ee_path"], float),
        [np.asarray(p, float) for p in episode["expert_phases"]],
        RECORD["phase_weights"],
        np.asarray(episode.get("speed", [0.0]), float),
        np.asarray(episode.get("forces", [0.0]), float),
        expert_total_len=float(episode.get("expert_total_len",
                                           RECORD["expert_total_len"])),
        t_exec=float(episode.get("t_exec", 1.0)),
        v_expert=float(episode.get("v_expert", 0.1)),
        success=bool(episode.get("success", False)),
        gamma_expert=RECORD["gamma_expert"],
        f_sus_lim=RECORD["f_sus_lim"],
        f_peak_lim=RECORD["f_peak_lim"])
'''


def synthesize(record: dict, llm=None, broken: bool = False) -> str:
    """Return Python source for the per-task scoring module."""
    if broken:
        return ("def compute_scores(episode):\n"
                "    return undefined_symbol  # fault injected for tests\n")
    if llm is not None:
        try:
            src = llm(
                "Write a self-contained Python module with "
                "compute_scores(episode)->dict returning keys "
                f"{sorted(_REQUIRED_KEYS)} for task "
                f"{record.get('task_description')!r}. Use "
                "robogym.metrics.paper_metrics.score_episode with these "
                f"constants: {record}. Output only code.")
            if "def compute_scores" in src:
                return src
        except Exception:
            pass
    return _baseline_scorer_source(record)


@dataclass
class RepairResult:
    source: str
    path: str
    rounds: int
    repaired: bool
    valid: bool
    error: str = ""
    sample_scores: dict = field(default_factory=dict)
    recovery: str = "first_try"


def _load(source: str, out_dir: Path) -> tuple[object, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"scoring_{uuid.uuid4().hex[:8]}.py"
    p.write_text(source)
    spec = importlib.util.spec_from_file_location(p.stem, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, str(p)


def _probe_episode(record: dict) -> dict:
    n = max(2, record.get("num_phases", 2))
    path = np.cumsum(np.ones((24, 3)) * 0.03, axis=0)
    return {
        "ee_path": path,
        "expert_phases": [path[: 12], *[path for _ in range(n - 1)]],
        "speed": np.full(24, 0.2),
        "forces": np.full(24, 8.0),
        "expert_total_len": record.get("expert_total_len", 1.0),
        "t_exec": 1.2, "v_expert": 0.3, "success": True,
    }


class CodeRepairLoop:
    """Synthesize, execute, and repair the scoring script until it validates."""

    def __init__(self, out_dir: str | Path = "results/scoring_scripts",
                 llm=None, max_rounds: int = 3):
        self.out_dir = Path(out_dir)
        self.llm = llm
        self.max_rounds = max_rounds

    def _validate(self, mod, record) -> tuple[bool, str, dict]:
        if not hasattr(mod, "compute_scores"):
            return False, "module has no compute_scores()", {}
        try:
            scores = mod.compute_scores(_probe_episode(record))
        except Exception:
            return False, traceback.format_exc(limit=3), {}
        missing = _REQUIRED_KEYS - set(scores or {})
        if missing:
            return False, f"missing keys: {sorted(missing)}", scores or {}
        return True, "", scores

    def run(self, record: dict, broken_first: bool = False) -> RepairResult:
        source = synthesize(record, llm=self.llm, broken=broken_first)
        last_err = ""
        used_baseline_fallback = False
        used_llm_repair = False
        for rnd in range(self.max_rounds):
            try:
                mod, path = _load(source, self.out_dir)
                ok, err, scores = self._validate(mod, record)
            except Exception:
                ok, err, scores, path = False, traceback.format_exc(
                    limit=3), {}, ""
            if ok:
                if rnd == 0:
                    recovery = "first_try"
                elif used_llm_repair and not used_baseline_fallback:
                    recovery = "llm_repair"
                else:
                    recovery = "baseline_fallback"
                return RepairResult(source, path, rnd, rnd > 0, True,
                                    "", scores, recovery)
            last_err = err
            if self.llm is not None:
                try:
                    source = self.llm(
                        f"This scoring script failed:\n{source}\n\n"
                        f"Error:\n{err}\n\nReturn a corrected full module.")
                    used_llm_repair = True
                    continue
                except Exception:
                    pass
            source = _baseline_scorer_source(record)
            used_baseline_fallback = True
        return RepairResult(source, "", self.max_rounds, True, False,
                            last_err, {}, "baseline_fallback")
