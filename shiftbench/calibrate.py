"""Fixed alarm cutoffs from a target false-alarm rate.

A *false alarm* is an alarm during an episode that would have succeeded. Given
the episode scores of n successful calibration episodes, the split-conformal
cutoff is the k-th smallest score with k = ceil((n + 1)(1 - alpha)). If a new
successful episode is exchangeable with the calibration episodes,

    P(new successful episode raises an alarm) <= alpha.

This is the promise the published detectors make (FAIL-Detect, Foresight,
FoMo-FD calibrate on successful rollouts the same way). It needs the new
episode to come from the same conditions as the calibration episodes. ShiftBench
measures what happens to it when the camera, lighting, table or objects change.

Ties: alarms fire on score > tau, so a tied score never alarms. That makes the
bound hold with ties too.
"""
from __future__ import annotations

import math

import numpy as np


def conformal_cutoff(success_scores, alpha: float) -> float:
    """Split-conformal cutoff. Returns +inf (never alarm) when n is too small,
    that is when alpha < 1 / (n + 1)."""
    s = np.sort(np.asarray(success_scores, float))
    n = len(s)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return float("inf")
    return float(s[k - 1])


def min_calibration_size(alpha: float) -> int:
    """Smallest n for which `conformal_cutoff` can return a finite cutoff."""
    return math.ceil(1 / alpha - 1)


def recalibrated_cutoff(new_success_scores, alpha: float, k: int) -> float:
    """Baseline: after a change, rerun the arm k times, keep the episodes that
    succeeded, and recompute the conformal cutoff on them alone. Valid for the
    new conditions, but costs k labeled rollouts and needs k >= 1/alpha - 1."""
    return conformal_cutoff(np.asarray(new_success_scores, float)[:k], alpha)
