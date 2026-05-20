"""Spearman ρ and Kendall's W (paper §4.3, Table 5 and §4.4, human study).

Computes the 5x5 inter-metric Spearman matrix (Table 5) and the inter-rater
Kendall's W / Spearman ρ for the §4.4 human-automation alignment study.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, spearmanr

def metric_correlation_matrix(model_table: dict[str, list],
                              metric_idx=(1, 2, 3, 4, 5)) -> np.ndarray:
    """5×5 Spearman ρ across models for [Comp,Space,Time,Smooth,Safety]
    (paper Table 5). ``model_table`` is ``{model: [SR,Comp,...,Safety]}``."""
    M = np.array([[row[i] for i in metric_idx]
                  for row in model_table.values()], float)
    k = M.shape[1]
    out = np.eye(k)
    for a in range(k):
        for b in range(a + 1, k):
            rho = spearmanr(M[:, a], M[:, b]).statistic
            out[a, b] = out[b, a] = 0.0 if np.isnan(rho) else rho
    return out

def kendalls_w(rankings: np.ndarray) -> float:
    """Kendall's coefficient of concordance W for ``(n_raters, n_items)``
    rankings. §4.4 inter-rater reliability."""
    R = np.asarray(rankings, float)
    m, n = R.shape                       # m raters, n items
    ranks = np.vstack([rankdata(r) for r in R])
    Rj = ranks.sum(axis=0)
    S = np.sum((Rj - Rj.mean()) ** 2)
    denom = m ** 2 * (n ** 3 - n)
    return float(12.0 * S / denom) if denom else 0.0

def spearman_vs_human(auto_scores: dict[str, float],
                      human_scores: dict[str, float]) -> float:
    """Spearman ρ between automated and human-consensus rankings (§4.4)."""
    keys = [k for k in auto_scores if k in human_scores]
    if len(keys) < 2:
        return 0.0
    a = [auto_scores[k] for k in keys]
    h = [human_scores[k] for k in keys]
    rho = spearmanr(a, h).statistic
    return 0.0 if np.isnan(rho) else float(rho)
