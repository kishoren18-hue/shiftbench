"""Failure scores that compare against successful runs instead of a world model.

The first offline run showed that world-model surprise barely separated failed
from successful rollouts (AUROC 0.41-0.61). These two scores test whether the
latents carry the signal at all, using a reference bank of successful episodes:

trajectory deviation (online, causal)
    At step t (time s_t seconds into the run), the distance from z_t to the
    nearest bank latent recorded within +-window_s seconds of the same time.
    "Does this run look like any good run did at this point?" Usable live, so it
    can drive the watchdog and the adaptive cutoff exactly like surprise.

end state (offline, after the run)
    Distance from the mean of the last few latents to the nearest bank end
    state. "Does the scene end the way good runs end?" Not an early warning; a
    baseline that says whether the encoder can see the outcome at all.

Latents are standardized and PCA-reduced on the bank (whitened, n_comp dims) so
distances are not dominated by a few large dimensions. The bank must never
contain the calibration or test episodes it scores.
"""
from __future__ import annotations

import numpy as np


class ReferenceBank:
    def __init__(self, episodes, n_comp: int = 32, window_s: float = 2.0, end_steps: int = 3):
        Z = np.concatenate([e.z for e in episodes]).astype(np.float64)
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-6
        Zs = (Z - self.mu) / self.sd
        _, S, Vt = np.linalg.svd(Zs - Zs.mean(0), full_matrices=False)
        k = min(n_comp, Vt.shape[0])
        self.P = Vt[:k].T / (S[:k] / np.sqrt(max(len(Zs) - 1, 1)) + 1e-6)   # whitening projection
        self.c = Zs.mean(0)
        self.window_s, self.end_steps = window_s, end_steps
        self.pts, self.times, self.ends = [], [], []
        for e in episodes:
            y = self.proj(e.z)
            self.pts.append(y)
            self.times.append(np.arange(len(y)) / e.fps)
            self.ends.append(y[-end_steps:].mean(0))
        self.pts = np.concatenate(self.pts)
        self.times = np.concatenate(self.times)
        self.ends = np.stack(self.ends)
        self.k = k

    def proj(self, z):
        return (((np.asarray(z, np.float64) - self.mu) / self.sd) - self.c) @ self.P

    def trajdev(self, e) -> np.ndarray:
        """Per-step distance to the nearest same-time bank latent (normalized by sqrt(k))."""
        y = self.proj(e.z)
        t = np.arange(len(y)) / e.fps
        out = np.empty(len(y))
        for i in range(len(y)):
            m = np.abs(self.times - t[i]) <= self.window_s
            cand = self.pts[m] if m.any() else self.pts
            out[i] = np.sqrt(((cand - y[i]) ** 2).sum(1).min() / self.k)
        return out

    def endstate(self, e) -> float:
        y = self.proj(e.z)[-self.end_steps:].mean(0)
        return float(np.sqrt(((self.ends - y) ** 2).sum(1).min() / self.k))
