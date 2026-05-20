"""Spectral Arc Length (SPARC) smoothness (paper §3.4, equation 5).

Uses the SPARC definition of Balasubramanian, Melendez-Calderon and
Burdet, IEEE TBME 2012 (paper reference [1]).

Equation (5)::

    S_SPARC = integral_0^{w_c}  sqrt( (d V_hat(w) / dw)^2  + 1 )  dw

with ``V_hat(w)`` the amplitude spectrum of the speed profile normalised by its
peak. All actions are projected into a standardized end-effector velocity space
``v_ee`` (§3.4). The arc length follows the ref-[1] sign convention (returned
negative, so closer to 0 => smoother). A Hann window is applied before the FFT,
and joint-space policies are mapped via the Jacobian ``v_ee = J(q) q_dot``.
"""

from __future__ import annotations

import numpy as np

def sparc(speed: np.ndarray, fs: float = 20.0, padlevel: int = 4,
          fc: float = 10.0, amp_th: float = 0.05) -> float:
    """Canonical Spectral Arc Length of a 1-D speed profile.

    Parameters
    ----------
    speed:
        Movement speed magnitude per timestep (e.g. ``||v_ee||``).
    fs:
        Sampling frequency (Hz; LIBERO control runs at 20 Hz).
    padlevel:
        Zero-pad to ``2**ceil(log2(N)) * 2**padlevel`` for spectral resolution.
    fc:
        Cutoff frequency ``w_c`` (Hz) for the arc-length integral.
    amp_th:
        Adaptive amplitude threshold for trimming the spectrum tail.

    Returns
    -------
    float
        SPARC value (negative; closer to 0 means smoother). Returns
        ``0.0`` for a degenerate (near-static) profile.
    """
    speed = np.asarray(speed, dtype=float).ravel()
    n = len(speed)
    if n < 4 or np.max(np.abs(speed)) < 1e-9:
        return 0.0

    # Hann window (standard SPARC practice) then zero-padded FFT
    speed = speed * np.hanning(n)
    nfft = int(2 ** (np.ceil(np.log2(n)) + padlevel))
    freq = np.arange(0, fs, fs / nfft)
    mag = np.abs(np.fft.fft(speed, nfft))
    mag = mag / (np.max(mag) + 1e-12)  # normalised amplitude spectrum V_hat

    # restrict to [0, fc]
    fc_idx = int(np.where(freq <= fc)[0][-1]) + 1 if np.any(freq <= fc) else len(freq)
    f_sel = freq[:fc_idx]
    m_sel = mag[:fc_idx]

    # adaptive trim to the informative band
    inx = np.where(m_sel >= amp_th)[0]
    if len(inx) >= 2:
        f_sel = f_sel[inx[0]: inx[-1] + 1]
        m_sel = m_sel[inx[0]: inx[-1] + 1]
    if len(m_sel) < 2:
        return 0.0

    # eq (5): arc length of the normalised amplitude spectrum.
    # Negative per the canonical ref-[1] convention; larger magnitude == more roughness.
    df = (f_sel[-1] - f_sel[0]) / (len(f_sel) - 1) if len(f_sel) > 1 else 1.0
    dmdf = np.diff(m_sel) / (df + 1e-12)
    arc = -np.sum(np.sqrt(dmdf ** 2 + 1.0)) * df
    return float(arc)

def jacobian_ee_velocity(jacobian: np.ndarray, qdot: np.ndarray) -> np.ndarray:
    """§3.4 EE-velocity projection: map joint velocities to EE cartesian
    velocity so heterogeneous action spaces are compared in one space.

    ``v_ee = J(q) @ qdot``. The Moore-Penrose pseudo-inverse ``J^+`` is
    used by callers that need the inverse map for redundant manipulators.
    """
    return np.asarray(jacobian, dtype=float) @ np.asarray(qdot, dtype=float)

def sparc_score(speed: np.ndarray, fs: float = 20.0,
                spread: float = 0.6) -> float:
    """Map raw SPARC to a calibrated ``[0, 100]`` smoothness score.

    SPARC is unbounded-negative; we map it monotonically so that 0 (perfectly
    smooth) maps to 100 and increasingly negative values decay toward 0,
    matching the direction and scale of the trajectory evaluator's
    smoothness score.
    """
    raw = sparc(speed, fs=fs)
    # raw is <= 0; -raw is "roughness". Gaussian decay keeps it in (0,100].
    return float(100.0 * np.exp(-((-raw) ** 2) / (2.0 * (spread ** 2))))
