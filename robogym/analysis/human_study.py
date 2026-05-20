"""Human-Automation Alignment study harness (paper §4.4).

§4.4 has three human experts rank 8 models across the five dimensions; the
reported headline is a mean Spearman ρ around 0.90 between the human
consensus and the automated-metric rankings. This harness registers experts,
collects per-expert per-dimension model rankings, persists them, and computes
Kendall's W and the per-dimension / mean Spearman ρ against the automated
rankings. :meth:`simulate_experts` is a *test harness only*: it perturbs the
automated ordering with controlled noise so the analysis path can be
exercised in CI without human input. The real study feeds real human
rankings into the same API via :meth:`add_expert_ranking`.
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
    """Collects expert rankings and aligns them with the automated metrics.

    The instance tracks a ``provenance`` flag (``"real_experts"`` after
    :meth:`add_expert_ranking` calls, ``"test_simulation"`` after
    :meth:`simulate_experts`). The flag is written into every report so
    downstream consumers can tell a real run apart from a harness run.
    """

    models: list[str]
    # rankings[expert][dim] = ordered list of models best -> worst
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
        """Test harness: populate placeholder rankings as the automated
        ordering plus bounded rank noise.

        This is not a substitute for the real study. Use
        :meth:`add_expert_ranking` to feed real human rankings. Calling
        this method sets :attr:`provenance` to ``"test_simulation"`` so the
        downstream report makes the source unambiguous.
        """
        rng = np.random.default_rng(seed)
        for e in range(n_experts):
            for di, dim in enumerate(DIMS):
                score = np.array([automated[m][di + 1] for m in self.models])
                jittered = score + rng.normal(0, noise * (score.std() + 1e-6),
                                              len(score))
                order = [self.models[i] for i in np.argsort(-jittered)]
                self.rankings.setdefault(f"expert_{e+1}", {})[dim] = list(order)
        self.provenance = "test_simulation"

    # analysis
    def _rank_matrix(self, dim: str) -> np.ndarray:
        """(n_experts, n_models) rank positions for ``dim``."""
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
        """Kendall's W per dimension + overall (§4.4 inter-rater reliability)."""
        out = {}
        for dim in DIMS:
            R = self._rank_matrix(dim)
            out[dim] = round(kendalls_w(R), 3) if len(R) >= 2 else None
        vals = [v for v in out.values() if v is not None]
        out["overall"] = round(float(np.mean(vals)), 3) if vals else None
        return out

    def consensus(self, dim: str) -> list[str]:
        """Borda-count human-consensus ranking for a dimension."""
        R = self._rank_matrix(dim)
        if len(R) == 0:
            return list(self.models)
        mean_rank = R.mean(axis=0)
        return [self.models[i] for i in np.argsort(mean_rank)]

    def spearman_vs_automated(self, automated: dict[str, list[float]]) -> dict:
        """Spearman ρ between human consensus and automated ranking per
        dimension, and the mean across dimensions (§4.4)."""
        out = {}
        for di, dim in enumerate(DIMS):
            cons = self.consensus(dim)
            human_score = {m: -cons.index(m) for m in self.models}
            auto_score = {m: automated[m][di + 1] for m in self.models}
            out[dim] = round(spearman_vs_human(auto_score, human_score), 3)
        out["mean"] = round(float(np.mean(list(out.values()))), 3)
        return out

    # persistence
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
        """Full §4.4 report: Kendall's W, per-dimension ρ, and mean ρ.

        The output file is named ``human_alignment_report.json`` for runs
        backed by real expert rankings and
        ``human_alignment_report_TEST_MODE.json`` when the rankings come
        from :meth:`simulate_experts`, so the two cannot be confused.
        """
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
