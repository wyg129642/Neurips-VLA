"""Human-automation alignment study harness (Sec. 4.4, Appendix B).

Three experts rank the eight models on the five dimensions; this harness
collects rankings, computes Kendall's W and Spearman rho versus the
automated metrics, and persists a report. :meth:`simulate_experts`
populates noisy stand-in rankings so the analysis path can run in CI; the
flag ``provenance`` distinguishes a simulated run from a real one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .correlation import kendalls_w, spearman_vs_human

DIMS = ["Comp", "Space", "Time", "Smooth", "Safety"]


@dataclass
class HumanStudy:
    models: list[str]
    rankings: dict = field(default_factory=dict)
    provenance: str = "uninitialised"

    def add_expert_ranking(self, expert: str, dim: str,
                           ordered_models: list[str]) -> None:
        assert dim in DIMS, dim
        assert set(ordered_models) == set(self.models), \
            "ranking must cover exactly the model set"
        self.rankings.setdefault(expert, {})[dim] = list(ordered_models)
        if self.provenance != "test_simulation":
            self.provenance = "real_experts"

    def simulate_experts(self, automated: dict[str, list[float]],
                         n_experts: int = 3, noise: float = 0.6,
                         seed: int = 0) -> None:
        """Populate stand-in rankings as automated ordering plus bounded noise."""
        rng = np.random.default_rng(seed)
        for e in range(n_experts):
            for di, dim in enumerate(DIMS):
                score = np.array([automated[m][di + 1] for m in self.models])
                jittered = score + rng.normal(0, noise * (score.std() + 1e-6),
                                              len(score))
                order = [self.models[i] for i in np.argsort(-jittered)]
                self.rankings.setdefault(f"expert_{e+1}", {})[dim] = list(order)
        self.provenance = "test_simulation"

    def _rank_matrix(self, dim: str) -> np.ndarray:
        idx = {m: i for i, m in enumerate(self.models)}
        rows = []
        for exp in sorted(self.rankings):
            order = self.rankings[exp].get(dim)
            if not order:
                continue
            r = np.zeros(len(self.models))
            for pos, m in enumerate(order):
                r[idx[m]] = pos + 1
            rows.append(r)
        return np.asarray(rows)

    def inter_rater_W(self) -> dict:
        out = {}
        for dim in DIMS:
            R = self._rank_matrix(dim)
            out[dim] = round(kendalls_w(R), 3) if len(R) >= 2 else None
        vals = [v for v in out.values() if v is not None]
        out["overall"] = round(float(np.mean(vals)), 3) if vals else None
        return out

    def consensus(self, dim: str) -> list[str]:
        R = self._rank_matrix(dim)
        if len(R) == 0:
            return list(self.models)
        mean_rank = R.mean(axis=0)
        return [self.models[i] for i in np.argsort(mean_rank)]

    def spearman_vs_automated(self, automated: dict[str, list[float]]) -> dict:
        out = {}
        for di, dim in enumerate(DIMS):
            cons = self.consensus(dim)
            human_score = {m: -cons.index(m) for m in self.models}
            auto_score = {m: automated[m][di + 1] for m in self.models}
            out[dim] = round(spearman_vs_human(auto_score, human_score), 3)
        out["mean"] = round(float(np.mean(list(out.values()))), 3)
        return out

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "models": self.models,
            "rankings": self.rankings,
            "provenance": self.provenance,
        }, indent=2))
        return p

    @classmethod
    def load(cls, path: str | Path) -> "HumanStudy":
        d = json.loads(Path(path).read_text())
        s = cls(models=d["models"])
        s.rankings = d["rankings"]
        s.provenance = d.get("provenance", "uninitialised")
        return s

    def report(self, automated: dict[str, list[float]],
               out_dir: str | Path) -> dict:
        rep = {
            "provenance": self.provenance,
            "n_experts": len(self.rankings),
            "kendalls_w": self.inter_rater_W(),
            "spearman_vs_automated": self.spearman_vs_automated(automated),
        }
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        name = ("human_alignment_report_TEST_MODE.json"
                if self.provenance == "test_simulation"
                else "human_alignment_report.json")
        (out / name).write_text(json.dumps(rep, indent=2))
        return rep
