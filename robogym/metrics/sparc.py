"""Spectral Arc Length (Sec. 3.4, Eq. 5).

Reference: Balasubramanian, Melendez-Calderon, Burdet, IEEE TBME 2012.
"""

from __future__ import annotations

import numpy as np


def sparc(speed: np.ndarray, fs: float = 20.0, padlevel: int = 4,
          fc: float = 10.0, amp_th: float = 0.05) -> float:
    """Arc length of the normalized amplitude spectrum, signed negative.

    Returns 0 for near-static speed profiles.
    """
    speed = np.asarray(speed, dtype=float).ravel()
    n = len(speed)
    if n < 4 or np.max(np.abs(speed)) < 1e-9:
        return 0.0

    windowed = speed * np.hanning(n)
    nfft = int(2 ** (np.ceil(np.log2(n)) + padlevel))
    freq = np.arange(0, fs, fs / nfft)
    mag = np.abs(np.fft.fft(windowed, nfft))
    mag = mag / (np.max(mag) + 1e-12)

    fc_mask = freq <= fc
    if not np.any(fc_mask):
        return 0.0
    fc_idx = int(np.where(fc_mask)[0][-1]) + 1
    f_sel = freq[:fc_idx]
    m_sel = mag[:fc_idx]

    inx = np.where(m_sel >= amp_th)[0]
    if len(inx) >= 2:
        f_sel = f_sel[inx[0]: inx[-1] + 1]
        m_sel = m_sel[inx[0]: inx[-1] + 1]
    if len(m_sel) < 2:
        return 0.0

    df = (f_sel[-1] - f_sel[0]) / (len(f_sel) - 1) if len(f_sel) > 1 else 1.0
    dmdf = np.diff(m_sel) / (df + 1e-12)
    return float(-np.sum(np.sqrt(dmdf ** 2 + 1.0)) * df)


def jacobian_ee_velocity(jacobian: np.ndarray, qdot: np.ndarray) -> np.ndarray:
    """Map joint-velocity commands to cartesian EE velocity, v_ee = J(q) q_dot."""
    return np.asarray(jacobian, dtype=float) @ np.asarray(qdot, dtype=float)


def sparc_score(speed: np.ndarray, fs: float = 20.0,
                spread: float = 2.5) -> float:
    """Map SPARC into a [0, 100] smoothness score with Gaussian decay."""
    raw = sparc(speed, fs=fs)
    return float(100.0 * np.exp(-((-raw) ** 2) / (2.0 * spread ** 2)))
