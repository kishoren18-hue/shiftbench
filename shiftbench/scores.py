"""Turning world-model prediction errors into one alarm score per moment.

The world model predicts the next latent frame from recent frames, the arm
state and the commanded action. Its error at step t is the *surprise*

    e_t = mean_d | z_hat_t[d] - z_t[d] |        (latents standardized per dim)

A single surprising frame is usually noise, so the alarm watches a smoothed
version, the mean surprise over the last `window` steps:

    s_t = mean(e_{t-window+1}, ..., e_t)

The alarm fires the first time s_t > tau. For calibration we only need one
number per episode, the *episode score* max_t s_t: an episode raises an alarm
under cutoff tau exactly when its episode score exceeds tau.

`reference` gives a self-referenced variant: subtract the median surprise of
the first `ref_steps` steps of the same episode. A camera bump that happened
before the episode starts raises every error in the episode by roughly the
same amount, so this variant cancels part of it. It cannot cancel a shift that
happens mid-episode, and it hides a failure that is already under way in the
first `ref_steps` steps. Both are tested in the benchmark, not assumed.
"""
from __future__ import annotations

import numpy as np


def surprise(z_pred: np.ndarray, z_true: np.ndarray) -> np.ndarray:
    """Per-step mean absolute error between predicted and observed latents.

    z_pred, z_true: (T, D) standardized latents. Returns (T,)."""
    z_pred = np.asarray(z_pred, float)
    z_true = np.asarray(z_true, float)
    if z_pred.shape != z_true.shape:
        raise ValueError(f"shape mismatch {z_pred.shape} vs {z_true.shape}")
    return np.abs(z_pred - z_true).mean(axis=-1)


def smooth(e: np.ndarray, window: int = 4) -> np.ndarray:
    """Trailing moving average with a shorter window at the start."""
    e = np.asarray(e, float)
    if window <= 1 or len(e) == 0:
        return e.copy()
    c = np.cumsum(np.r_[0.0, e])
    idx = np.arange(1, len(e) + 1)
    lo = np.maximum(idx - window, 0)
    return (c[idx] - c[lo]) / (idx - lo)


def reference(e: np.ndarray, ref_steps: int = 4) -> np.ndarray:
    """Surprise relative to the episode's own first `ref_steps` steps."""
    e = np.asarray(e, float)
    if len(e) == 0:
        return e.copy()
    return e - np.median(e[: max(1, ref_steps)])


def score_trace(e: np.ndarray, window: int = 4, ref_steps: int | None = None) -> np.ndarray:
    """Alarm statistic s_t for one episode. With ref_steps, the first
    ref_steps - 1 steps are -inf (no alarm until the reference exists), so the
    offline trace equals what the live scorer can compute."""
    if ref_steps:
        tr = smooth(reference(e, ref_steps), window)
        tr[: ref_steps - 1] = -np.inf
        return tr
    return smooth(e, window)


def episode_score(trace: np.ndarray) -> float:
    """Largest alarm statistic in the episode (-inf for empty episodes)."""
    trace = np.asarray(trace, float)
    return float(trace.max()) if len(trace) else float("-inf")


def first_alarm(trace: np.ndarray, tau: float) -> int | None:
    """Index of the first step whose statistic exceeds tau, or None."""
    hits = np.flatnonzero(np.asarray(trace) > tau)
    return int(hits[0]) if len(hits) else None
