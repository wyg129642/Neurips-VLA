"""Spearman rho and Kendall's W (Tables 5-6)."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, spearmanr


def metric_correlation_matrix(model_table: dict[str, list],
                              metric_idx=(1, 2, 3, 4, 5)) -> np.ndarray:
    """5x5 Spearman rho across models for [Comp, Space, Time, Smooth, Safety]."""
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
    """Kendall's W for ``(n_raters, n_items)`` rank matrix."""
    R = np.asarray(rankings, float)
    m, n = R.shape
    ranks = np.vstack([rankdata(r) for r in R])
    Rj = ranks.sum(axis=0)
    S = np.sum((Rj - Rj.mean()) ** 2)
    denom = m ** 2 * (n ** 3 - n)
    return float(12.0 * S / denom) if denom else 0.0


def spearman_vs_human(auto_scores: dict[str, float],
                      human_scores: dict[str, float]) -> float:
    keys = [k for k in auto_scores if k in human_scores]
    if len(keys) < 2:
        return 0.0
    a = [auto_scores[k] for k in keys]
    h = [human_scores[k] for k in keys]
    rho = spearmanr(a, h).statistic
    return 0.0 if np.isnan(rho) else float(rho)
